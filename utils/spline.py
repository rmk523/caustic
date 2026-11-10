import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import struct

class DifferentiableBspline(nn.Module):
    def __init__(self, num_ctrl_u, num_ctrl_v, degree=3, 
                 x_range=(0, 40), y_range=(0, 40), padding=0.01, avg_height=0):
        super().__init__()
        
        self.p = degree
        self.num_u = num_ctrl_u
        self.num_v = num_ctrl_v
        self.x_range = x_range
        self.y_range = y_range
        self.padding = padding

        self.ctrl_pts = nn.Parameter(torch.randn(self.num_u, self.num_v) * 0.01 + avg_height)

        self.register_buffer('knots_u', self._generate_knots(num_ctrl_u, degree, x_range, padding))
        self.register_buffer('knots_v', self._generate_knots(num_ctrl_v, degree, y_range, padding))

        self.refine_surface_matrix = self._build_refine_surface_matrix()

    def _build_refine_surface_matrix(self):
        refine_surface_matrix_np = np.zeros((self.p+1, self.p+2), dtype=np.float32)
        for i in range(1, self.p+2):
            L = self.p + 2
            out = np.zeros(L, dtype=np.float32)
            out[-i:] = np.arange(1, i+1, dtype=np.float32)

            refined_nonzero = np.arange(1, 2*i + 1, dtype=np.float32) * 0.5
            num_zeros = self.p + 1
            zeros_part = np.zeros(num_zeros, dtype=np.float32)
            final_out = np.concatenate((zeros_part, refined_nonzero))
            max_len = self.p * 2 + 3
            final_out = final_out[-max_len:]

            refine_result = self.solve_refinement_coefficients(self.p, i, out, final_out)

            result_len = len(refine_result)
            target_len = self.p + 2
            if result_len < target_len:
                refine_result = np.concatenate([
                    np.zeros(target_len - result_len, dtype=np.float32),
                    refine_result
                ])
            else:
                refine_result = refine_result[-target_len:]

            refine_surface_matrix_np[i-1, :] = refine_result
            refine_surface_matrix_np[np.abs(refine_surface_matrix_np) < 1e-10] = 0.0
            refine_surface_matrix = torch.from_numpy(refine_surface_matrix_np)
        return refine_surface_matrix
    
    def _generate_knots(self, num_ctrl, p, val_range, padding):
        min_val, max_val = val_range
        span = max_val - min_val
        
        # 实际内部节点覆盖范围 (扩大一点以处理边界数据)
        padded_min = min_val - span * padding
        padded_max = max_val + span * padding
        
        # 内部节点数量 = num_ctrl - p
        inner_knots = torch.linspace(padded_min, padded_max, num_ctrl - p + 1)
        
        knots = torch.cat([
            inner_knots[0].repeat(p), # Start multiplicity
            inner_knots,
            inner_knots[-1].repeat(p) # End multiplicity
        ])
        return knots

    def _find_span(self, u, knots):
        span_idx = torch.searchsorted(knots, u, side='right') - 1
        
        # 限制索引防止越界 (处理数值误差)
        max_idx = len(knots) - self.p - 2
        span_idx = torch.clamp(span_idx, self.p, max_idx)
        return span_idx
    
    def eval_specific_basis(self, u, knots):
        p = knots.shape[0] - 2   # 阶数 p

        # -------- 1. 基础 0 阶基函数 --------
        knots_left  = knots[:-1]
        knots_right = knots[1:]

        # (n,1) 和 (1,p+1) 自动广播得到 (n,p+1)
        u_col = u[:, None]

        buffer = ((u_col >= knots_left) & (u_col < knots_right)).astype(float)

        # -------- 2. Cox-de Boor 递推 --------
        for d in range(1, p + 1):
            denom_left = knots[d:-1] - knots[:-d-1]          # (p+1-d,)
            term_left = ((u_col - knots[:-d-1]) / (denom_left + 1e-12)) * buffer[:, :-1]
            denom_right = knots[d+1:] - knots[1:-d]          # (p+1-d,)
            term_right = ((knots[d+1:] - u_col) / (denom_right + 1e-12)) * buffer[:, 1:]
            buffer = term_left + term_right

        return buffer[:, 0]
    
    def solve_refinement_coefficients(self, p, i, out, final_out):
        num_fine_basis = len(final_out) - p - 1

        edges = np.linspace(0, i, num_fine_basis + 2)
        u_samples = edges[1:-1]

        mat_A = np.zeros((len(u_samples), num_fine_basis))

        for j in range(num_fine_basis):
            local_knots = final_out[j : j + p + 2]
            
            # 计算该细分基函数在所有 u 处的值
            basis_vals = self.eval_specific_basis(u_samples, local_knots)
            mat_A[:, j] = basis_vals

        vec_b = self.eval_specific_basis(u_samples, out)

        coefficients, residuals, rank, s = np.linalg.lstsq(mat_A, vec_b, rcond=None)

        return coefficients
    
    def _basis_funs_batch(self, u, span_idxs, knots):
            """
            计算非零基函数的值 (Cox-de Boor 递归) - 完全 Autograd 安全版
            """
            p = self.p
            batch_size = u.shape[0]
            device = u.device
            dtype = u.dtype
            
            N = torch.ones(batch_size, 1, device=device, dtype=dtype)
            
            left = [None] * (p + 1)
            right = [None] * (p + 1)
            
            for j in range(1, p + 1):
                left[j] = u - knots[span_idxs + 1 - j]
                right[j] = knots[span_idxs + j] - u
                
                saved = torch.zeros(batch_size, device=device, dtype=dtype)
                
                # 暂存下一阶的 N 值
                N_next_list = []
                
                for r in range(j):
                    term = N[:, r] / (right[r + 1] + left[j - r] + 1e-8)
                    
                    val = saved + right[r + 1] * term
                    N_next_list.append(val)
                    
                    saved = left[j - r] * term
                
                # 补上最后一个值
                N_next_list.append(saved)
                
                # 生成新的 N Tensor，保留旧 N 用于梯度回传
                N = torch.stack(N_next_list, dim=1)
                
            return N

    def forward(self, uv_coords):
        u = uv_coords[:, 0].contiguous()
        v = uv_coords[:, 1].contiguous()
        
        span_u = self._find_span(u, self.knots_u)
        span_v = self._find_span(v, self.knots_v)
        
        basis_u = self._basis_funs_batch(u, span_u, self.knots_u)
        basis_v = self._basis_funs_batch(v, span_v, self.knots_v)
        
        idx_u = torch.stack([span_u - self.p + k for k in range(self.p+1)], dim=1) # (Batch, p+1)
        idx_v = torch.stack([span_v - self.p + k for k in range(self.p+1)], dim=1) # (Batch, p+1)

        local_ctrl = self.ctrl_pts[idx_u[:, :, None], idx_v[:, None, :]]

        basis_tensor = basis_u[:, :, None] * basis_v[:, None, :] 
        
        numerator = torch.sum(basis_tensor * local_ctrl, dim=(1, 2))

        z = numerator
        
        return z[:, None] 

    def forward_precomputed_ckpt(self, precomputed_regular_data):
        out = self.forward_precomputed_2(precomputed_regular_data)
        return out["z"], out["grads"], out["grads_2nd"]
    
    def compute_first_derivatives(self, uv_points):
        """
        计算 z(u,v) 和一阶导数 z_u, z_v
        """
        uv_points = uv_points.clone().detach().requires_grad_(True)

        # z(u,v)
        z = self(uv_points)
        if z.ndim > 1:
            z = z.squeeze(-1)

        # 一阶导数
        grads = torch.autograd.grad(
            outputs=z,
            inputs=uv_points,
            grad_outputs=torch.ones_like(z),
            create_graph=True
        )[0]

        return {
            "z": z,
            "grads": grads
        }


    def compute_second_derivatives(self, uv_points):

        uv_points.requires_grad_(True)

        z = self(uv_points)
        if z.ndim > 1:
            z = z.squeeze(-1)

        grads = torch.autograd.grad(
            outputs=z,
            inputs=uv_points,
            grad_outputs=torch.ones_like(z),
            create_graph=True
        )[0]

        z_u = grads[:, 0]
        z_v = grads[:, 1]

        grads_x = torch.autograd.grad(
            outputs=z_u,
            inputs=uv_points,
            grad_outputs=torch.ones_like(z_u),
            create_graph=True
        )[0]

        grads_y = torch.autograd.grad(
            outputs=z_v,
            inputs=uv_points,
            grad_outputs=torch.ones_like(z_v),
            create_graph=True
        )[0]

        return {
            "z": z,
            "grads": grads,
            "grads_x": grads_x, "grads_y": grads_y,
        }
    
    def refine_surface(self):
        current_ctrl = self.ctrl_pts
        device, dtype = current_ctrl.device, current_ctrl.dtype
        p = self.p
        
        def get_transform_mat(n):
            width = 2 * n + p
            mat = torch.zeros(n, width, device=device, dtype=dtype)
            
            refine_mat = self.refine_surface_matrix.to(device=device, dtype=dtype)
            kernel_uniform = refine_mat[-1]
            
            for i in range(n):
                col_start = 2 * i
                col_end = col_start + p + 2
                if i < p:
                    coeffs = refine_mat[i]
                elif i >= n - p:
                    idx_mirror = (n - 1) - i
                    coeffs = torch.flip(refine_mat[idx_mirror], dims=[0])
                else:
                    coeffs = kernel_uniform

                # 填充矩阵
                if col_end <= width:
                    mat[i, col_start : col_end] = coeffs
                else:
                    valid_len = width - col_start
                    mat[i, col_start : width] = coeffs[:valid_len]
                
            return mat

        Mat_u = get_transform_mat(self.num_u)  # Shape: (Nu, 2*Nu + p)
        Mat_v = get_transform_mat(self.num_v)  # Shape: (Nv, 2*Nv + p)

        new_ctrl_full = Mat_u.T @ current_ctrl @ Mat_v

        final_ctrl = new_ctrl_full[p:-p, p:-p]

        self.num_u = 2 * self.num_u - p
        self.num_v = 2 * self.num_v - p

        self.knots_u = self._generate_knots(self.num_u, self.p, self.x_range, self.padding).to(device)
        self.knots_v = self._generate_knots(self.num_v, self.p, self.y_range, self.padding).to(device)

        self.ctrl_pts = nn.Parameter(final_ctrl)
        
        self.is_precomputed = False
        
        return

    def precompute_basis_data_1(self, uv_coords):
        u = uv_coords[:, 0].clone().detach().requires_grad_(True)
        v = uv_coords[:, 1].clone().detach().requires_grad_(True)
        
        with torch.no_grad():
            span_u = self._find_span(u, self.knots_u)
            span_v = self._find_span(v, self.knots_v)
            
            idx_u = torch.stack([span_u - self.p + k for k in range(self.p+1)], dim=1).long()
            idx_v = torch.stack([span_v - self.p + k for k in range(self.p+1)], dim=1).long()

        basis_u_val = self._basis_funs_batch(u, span_u, self.knots_u)
        basis_v_val = self._basis_funs_batch(v, span_v, self.knots_v)

        grad_u_list = []
        for k in range(self.p + 1):
            grad_k = torch.autograd.grad(
                outputs=basis_u_val[:, k],
                inputs=u,
                grad_outputs=torch.ones_like(u),
                create_graph=False, # 预计算不需要保留图
                retain_graph=True
            )[0]
            grad_u_list.append(grad_k)
        basis_u_grad = torch.stack(grad_u_list, dim=1)

        # --- V 方向导数 ---
        grad_v_list = []
        for k in range(self.p + 1):
            grad_k = torch.autograd.grad(
                outputs=basis_v_val[:, k],
                inputs=v,
                grad_outputs=torch.ones_like(v),
                create_graph=False,
                retain_graph=True
            )[0]
            grad_v_list.append(grad_k)
        basis_v_grad = torch.stack(grad_v_list, dim=1)

        with torch.no_grad():
            term_base = basis_u_val[:, :, None] * basis_v_val[:, None, :]
            term_du = basis_u_grad[:, :, None] * basis_v_val[:, None, :]
            term_dv = basis_u_val[:, :, None] * basis_v_grad[:, None, :]

        precomputed_data = {
            "idx_u": idx_u.detach(),          # (N, p+1)
            "idx_v": idx_v.detach(),          # (N, p+1)
            "term_base": term_base.detach(),  # (N, p+1, p+1)
            "term_du": term_du.detach(),      # (N, p+1, p+1)
            "term_dv": term_dv.detach()       # (N, p+1, p+1)
        }
        
        return precomputed_data

    def precompute_basis_data_2(self, uv_coords):
        u = uv_coords[:, 0].clone().detach().requires_grad_(True)
        v = uv_coords[:, 1].clone().detach().requires_grad_(True)
        
        with torch.no_grad():
            span_u = self._find_span(u, self.knots_u)
            span_v = self._find_span(v, self.knots_v)
            idx_u = torch.stack([span_u - self.p + k for k in range(self.p+1)], dim=1).long()
            idx_v = torch.stack([span_v - self.p + k for k in range(self.p+1)], dim=1).long()

        basis_u_val = self._basis_funs_batch(u, span_u, self.knots_u)
        basis_v_val = self._basis_funs_batch(v, span_v, self.knots_v)
        
        grad_u_list = []
        for k in range(self.p + 1):
            grad_k = torch.autograd.grad(
                outputs=basis_u_val[:, k],
                inputs=u,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            grad_u_list.append(grad_k)
        basis_u_grad = torch.stack(grad_u_list, dim=1) # (N, p+1)
        
        grad_u_list2 = []
        for k in range(self.p + 1):
            grad_k2 = torch.autograd.grad(
                outputs=basis_u_grad[:, k],
                inputs=u,
                grad_outputs=torch.ones_like(u),
                create_graph=False, # 二阶导数预计算完成，不需要保留图
                retain_graph=True
            )[0]
            grad_u_list2.append(grad_k2)
        basis_u_grad2 = torch.stack(grad_u_list2, dim=1) # (N, p+1)

        grad_v_list = []
        for k in range(self.p + 1):
            grad_k = torch.autograd.grad(
                outputs=basis_v_val[:, k],
                inputs=v,
                grad_outputs=torch.ones_like(v),
                create_graph=True, # **关键：用于计算二阶导数**
                retain_graph=True
            )[0]
            grad_v_list.append(grad_k)
        basis_v_grad = torch.stack(grad_v_list, dim=1) # (N, p+1)

        grad_v_list2 = []
        for k in range(self.p + 1):
            grad_k2 = torch.autograd.grad(
                outputs=basis_v_grad[:, k],
                inputs=v,
                grad_outputs=torch.ones_like(v),
                create_graph=False,
                retain_graph=True
            )[0]
            grad_v_list2.append(grad_k2)
        basis_v_grad2 = torch.stack(grad_v_list2, dim=1) # (N, p+1)
        
        with torch.no_grad():
            term_base = basis_u_val[:, :, None] * basis_v_val[:, None, :]        
            term_du = basis_u_grad[:, :, None] * basis_v_val[:, None, :]
            term_dv = basis_u_val[:, :, None] * basis_v_grad[:, None, :]
            
            term_uu = basis_u_grad2[:, :, None] * basis_v_val[:, None, :]
            term_uv = basis_u_grad[:, :, None] * basis_v_grad[:, None, :]
            term_vv = basis_u_val[:, :, None] * basis_v_grad2[:, None, :]

        precomputed_data = {
            "idx_u": idx_u.detach(),             # (N, p+1)
            "idx_v": idx_v.detach(),             # (N, p+1)
            "term_base": term_base.detach(),     # (N, p+1, p+1)
            "term_du": term_du.detach(),         # (N, p+1, p+1)
            "term_dv": term_dv.detach(),         # (N, p+1, p+1)
            "term_uu": term_uu.detach(),         # (N, p+1, p+1)
            "term_uv": term_uv.detach(),         # (N, p+1, p+1)
            "term_vv": term_vv.detach()          # (N, p+1, p+1)
        }
        
        return precomputed_data

    def forward_precomputed_1(self, precomputed_data):
        # 1. 取出数据
        idx_u = precomputed_data["idx_u"]
        idx_v = precomputed_data["idx_v"]
        term_base = precomputed_data["term_base"]
        term_du = precomputed_data["term_du"]
        term_dv = precomputed_data["term_dv"]

        local_ctrl = self.ctrl_pts[idx_u[:, :, None], idx_v[:, None, :]]

        z = torch.sum(local_ctrl * term_base, dim=(1, 2))
        z_u = torch.sum(local_ctrl * term_du, dim=(1, 2))
        z_v = torch.sum(local_ctrl * term_dv, dim=(1, 2))

        grads = torch.stack([z_u, z_v], dim=1)

        return {
            "z": z[:, None],       # (Batch, )
            "grads": grads # (Batch, 2)
        }
    
    def forward_precomputed_2(self, precomputed_data):
        idx_u = precomputed_data["idx_u"]
        idx_v = precomputed_data["idx_v"]
        term_base = precomputed_data["term_base"]
        term_du = precomputed_data["term_du"]
        term_dv = precomputed_data["term_dv"]
        term_uu = precomputed_data["term_uu"]
        term_uv = precomputed_data["term_uv"]
        term_vv = precomputed_data["term_vv"]

        local_ctrl = self.ctrl_pts[idx_u[:, :, None], idx_v[:, None, :]]

        z = torch.sum(local_ctrl * term_base, dim=(1, 2))
        z_u = torch.sum(local_ctrl * term_du, dim=(1, 2))
        z_v = torch.sum(local_ctrl * term_dv, dim=(1, 2))
        
        z_uu = torch.sum(local_ctrl * term_uu, dim=(1, 2))
        z_uv = torch.sum(local_ctrl * term_uv, dim=(1, 2))
        z_vv = torch.sum(local_ctrl * term_vv, dim=(1, 2))

        grads = torch.stack([z_u, z_v], dim=1) # (Batch, 2)
        grads_2nd = torch.stack([z_uu, z_uv, z_vv], dim=1) # (Batch, 3)

        return {
            "z": z[:, None],              # (Batch, )
            "grads": grads, # (Batch, 2)
            "grads_2nd": grads_2nd # (Batch, 3)
        }

    def forward_precomputed_2_check(self, precomputed_data):
        def check_bad_values(name, x, max_print=20):
            if not torch.is_tensor(x):
                return False

            bad_mask = torch.isnan(x) | torch.isinf(x)

            if bad_mask.any():
                idx = torch.nonzero(bad_mask, as_tuple=False)
                print(f"\n{name} has bad values:")
                print(f"  total bad values: {idx.shape[0]}")

                for i in idx[:max_print]:
                    i_tuple = tuple(i.tolist())
                    val = x[i_tuple]
                    if torch.isnan(val):
                        typ = "NaN"
                    elif torch.isinf(val):
                        typ = "Inf"
                    else:
                        typ = "Unknown"

                    print(f"  {name}{i_tuple} = {val.item()} ({typ})")

                return True

            return False

        has_bad = False

        idx_u = precomputed_data["idx_u"]
        idx_v = precomputed_data["idx_v"]
        term_base = precomputed_data["term_base"]
        term_du = precomputed_data["term_du"]
        term_dv = precomputed_data["term_dv"]
        term_uu = precomputed_data["term_uu"]
        term_uv = precomputed_data["term_uv"]
        term_vv = precomputed_data["term_vv"]

        for name, tensor in {
            "self.ctrl_pts": self.ctrl_pts,
            "idx_u": idx_u,
            "idx_v": idx_v,
            "term_base": term_base,
            "term_du": term_du,
            "term_dv": term_dv,
            "term_uu": term_uu,
            "term_uv": term_uv,
            "term_vv": term_vv,
        }.items():
            has_bad = check_bad_values(name, tensor) or has_bad

        local_ctrl = self.ctrl_pts[idx_u[:, :, None], idx_v[:, None, :]]
        has_bad = check_bad_values("local_ctrl", local_ctrl) or has_bad

        z = torch.sum(local_ctrl * term_base, dim=(1, 2))
        z_u = torch.sum(local_ctrl * term_du, dim=(1, 2))
        z_v = torch.sum(local_ctrl * term_dv, dim=(1, 2))

        z_uu = torch.sum(local_ctrl * term_uu, dim=(1, 2))
        z_uv = torch.sum(local_ctrl * term_uv, dim=(1, 2))
        z_vv = torch.sum(local_ctrl * term_vv, dim=(1, 2))

        for name, tensor in {
            "z": z,
            "z_u": z_u,
            "z_v": z_v,
            "z_uu": z_uu,
            "z_uv": z_uv,
            "z_vv": z_vv,
        }.items():
            has_bad = check_bad_values(name, tensor) or has_bad

        grads = torch.stack([z_u, z_v], dim=1)
        grads_2nd = torch.stack([z_uu, z_uv, z_vv], dim=1)

        for name, tensor in {
            "grads": grads,
            "grads_2nd": grads_2nd,
            "z[:, None]": z[:, None],
        }.items():
            has_bad = check_bad_values(name, tensor) or has_bad

        if has_bad:
            print("\nforward_precomputed_2_check result: False")
        else:
            print("\nforward_precomputed_2_check result: True")

        return {
            "z": z[:, None],
            "grads": grads,
            "grads_2nd": grads_2nd,
            "is_valid": not has_bad,
        }
        
    def load_spline(self, path, device="cpu", strict_degree=True):
        # 1) torch.load 没有 device 参数，只有 map_location
        ckpt = torch.load(path, map_location="cpu")

        if not isinstance(ckpt, DifferentiableBspline):
            raise TypeError(f"Expected DifferentiableBspline, got {type(ckpt)}")

        # 2) degree 检查（你说 self.p 不变）
        if strict_degree and hasattr(ckpt, "p") and self.p != ckpt.p:
            raise ValueError(f"Degree mismatch: self.p={self.p}, ckpt.p={ckpt.p}")

        # 3) 从 ckpt 拿出数据（先在 CPU 上 clone，避免引用）
        ctrl_pts = ckpt.ctrl_pts.detach().clone()
        knots_u  = ckpt.knots_u.detach().clone()
        knots_v  = ckpt.knots_v.detach().clone()
        rsm      = ckpt.refine_surface_matrix.detach().clone()

        # 4) 根据 knots 长度反推 num_u / num_v
        # len(knots) = num_ctrl + degree + 1
        self.num_u = int(knots_u.numel() - self.p - 1)
        self.num_v = int(knots_v.numel() - self.p - 1)

        # 5) 覆盖 Parameter（比 copy_ 稳：允许形状变化）
        self.ctrl_pts = nn.Parameter(ctrl_pts)

        # 6) 覆盖 buffers（用 register_buffer，保证 .to(device) 会搬）
        def _set_buffer(self, name, tensor):
            # tensor: CPU tensor is fine; we'll .to(device) later
            if name in self._buffers:
                if self._buffers[name].shape == tensor.shape:
                    self._buffers[name].copy_(tensor)
                    return
                # shape 不同：必须删掉旧的再注册
                del self._buffers[name]
            if hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, tensor)
            
        with torch.no_grad():
            _set_buffer(self, "knots_u", knots_u)
            _set_buffer(self, "knots_v", knots_v)
            _set_buffer(self, "refine_surface_matrix", rsm)

        # 7) 最后整体搬到目标 device
        self.to(device)

        # 8) sanity check
        if tuple(self.ctrl_pts.shape) != (self.num_u, self.num_v):
            raise ValueError(
                f"ctrl_pts shape {tuple(self.ctrl_pts.shape)} "
                f"!= inferred (num_u,num_v)=({self.num_u},{self.num_v})"
            )
            
    def export_bspline_to_stl(self, z_back, filename, res_u=150, res_v=150):
        """
        将 DifferentiableBspline 曲面导出为封闭的 STL 实体。
        所有侧面的法线均指向外部。
        """
        # 获取模型所在设备
        device = next(self.parameters()).device

        # 使用类属性中的物理范围采样
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        u_vals = torch.linspace(x0, x1, res_u, device=device)
        v_vals = torch.linspace(y0, y1, res_v, device=device)
        uv_grid = torch.stack(torch.meshgrid(u_vals, v_vals, indexing='ij'), dim=-1).reshape(-1, 2)

        with torch.no_grad():
            z_vals = self(uv_grid).squeeze().cpu().numpy()
        z_vals = z_vals.reshape(res_u, res_v)

        x_coords = np.linspace(x0, x1, res_u)
        y_coords = np.linspace(y0, y1, res_v)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        front_verts = np.stack([X, Y, z_vals], axis=-1)
        back_verts = np.stack([X, Y, np.full_like(z_vals, z_back)], axis=-1)

        all_verts = np.vstack([front_verts.reshape(-1, 3), back_verts.reshape(-1, 3)])
        if not np.all(np.isfinite(all_verts)):
            print("警告：顶点数据包含 NaN 或 Inf，将替换为 0")
            all_verts = np.nan_to_num(all_verts, nan=0.0, posinf=0.0, neginf=0.0)

        faces = []

        def add_quad(v1, v2, v3, v4):
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

        # ----- 前表面 (法线朝 +Z) -----
        for i in range(res_u - 1):
            for j in range(res_v - 1):
                idx = i * res_v + j
                add_quad(idx, idx + res_v, idx + res_v + 1, idx + 1)

        # ----- 后表面 (法线朝 -Z) -----
        offset = res_u * res_v
        for i in range(res_u - 1):
            for j in range(res_v - 1):
                idx = i * res_v + j + offset
                faces.append([idx, idx + 1, idx + res_v + 1])
                faces.append([idx, idx + res_v + 1, idx + res_v])

        # ----- 侧面生成 (显式指定顶点顺序，确保法线向外) -----
        # 边界索引
        front_edges = {
            'u0': [0 * res_v + j for j in range(res_v)],           # x = x0
            'u1': [(res_u - 1) * res_v + j for j in range(res_v)], # x = x1
            'v0': [i * res_v + 0 for i in range(res_u)],           # y = y0
            'v1': [i * res_v + (res_v - 1) for i in range(res_u)]  # y = y1
        }
        back_edges = {k: [idx + offset for idx in front_edges[k]] for k in front_edges}

        # 边 u=0 (x最小，外法线 -x)
        for k in range(res_v - 1):
            f1 = front_edges['u0'][k]
            f2 = front_edges['u0'][k+1]
            b1 = back_edges['u0'][k]
            b2 = back_edges['u0'][k+1]
            faces.append([f1, f2, b2])
            faces.append([f1, b2, b1])

        # 边 u=1 (x最大，外法线 +x)
        for k in range(res_v - 1):
            f1 = front_edges['u1'][k]
            f2 = front_edges['u1'][k+1]
            b1 = back_edges['u1'][k]
            b2 = back_edges['u1'][k+1]
            faces.append([f1, b1, b2])
            faces.append([f1, b2, f2])

        # 边 v=0 (y最小，外法线 -y)
        for k in range(res_u - 1):
            f1 = front_edges['v0'][k]
            f2 = front_edges['v0'][k+1]
            b1 = back_edges['v0'][k]
            b2 = back_edges['v0'][k+1]
            faces.append([f1, b1, b2])
            faces.append([f1, b2, f2])

        # 边 v=1 (y最大，外法线 +y)
        for k in range(res_u - 1):
            f1 = front_edges['v1'][k]
            f2 = front_edges['v1'][k+1]
            b1 = back_edges['v1'][k]
            b2 = back_edges['v1'][k+1]
            faces.append([f1, f2, b2])
            faces.append([f1, b2, b1])

        # ----- 写入二进制 STL -----
        import struct
        with open(filename, 'wb') as f:
            f.write(b'\x00' * 80)
            num_faces = len(faces)
            f.write(struct.pack('<I', num_faces))

            for tri in faces:
                v0 = all_verts[tri[0]]
                v1 = all_verts[tri[1]]
                v2 = all_verts[tri[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    normal = np.array([0.0, 0.0, 1.0])
                else:
                    normal = normal / norm
                f.write(struct.pack('<fff', normal[0], normal[1], normal[2]))
                for idx in tri:
                    f.write(struct.pack('<fff', all_verts[idx][0], all_verts[idx][1], all_verts[idx][2]))
                f.write(struct.pack('<H', 0))

        print(f"STL 已导出到 {filename}，包含 {num_faces} 个三角形")