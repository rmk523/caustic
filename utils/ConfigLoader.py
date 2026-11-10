import os
import json
import shutil

# Repository root, derived from this file's location (utils/ConfigLoader.py).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class sp_JSONDataProcessor:
    def __init__(self, folder_path: str, encoding: str = 'utf-8'):
        self.root_dir = ROOT_DIR
        self.folder_path = os.path.join(self.root_dir, folder_path)
        self.encoding = encoding
        self.data = None
        self._load_data()
        self._path_set()
        # self.data_deal()

    def _load_data(self):
        try:
            with open(self.folder_path, 'r', encoding=self.encoding) as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"JSON文件 {self.folder_path} 不存在")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {str(e)}")
            
    def copy_json_to_targets(self, target_folder):
        os.makedirs(target_folder, exist_ok=True)
        original_name = os.path.basename(self.folder_path)
        json_target = os.path.join(target_folder, original_name)
        shutil.copy2(self.folder_path, json_target)

    def _path_set(self):
        # 输入图片
        self.data['image_path'] = self.root_dir+"/data/pic_data/{}.{}".format(self.data['picture_para']['name'],self.data['picture_para']['type'])
        self.data['image_txt'] = self.root_dir+"/data/pic_data/{}.{}".format(self.data['picture_para']['name'],"txt")
        # 输入半程数据 
        input_name = self.data['input_point_para']['input_data_name']
        iter_num = str(self.data['input_point_para']['input_data_iter_number'])
        data_num = str(self.data['input_point_para']['input_data_number'])
        self.data['aim_xy_data_path'] = (f"{self.root_dir}/result_sp_data/"f"{input_name}/data_point/{iter_num}/aim_xy_data_{data_num}.pt")
        self.data['origin_xy_data_path'] = (f"{self.root_dir}/result_sp_data/"f"{input_name}/data_point/{iter_num}/origin_xy_data_{data_num}.pt")
        self.data['triangle_tensor_path'] = (f"{self.root_dir}/result_sp_data/"f"{input_name}/data_point/{iter_num}/triangle_tensor_{data_num}.pt")
        # 所用修饰词
        self.data['xiushi'] = self.data['sample_point_para']['data_name']+"_"+self.data['test_num']
        # 采样点数据储存位置
        self.data['data_point_folder'] = self.root_dir+"/result_sp_data/"+self.data["xiushi"]+"/data_point/"
        self.data['data_mesh_folder'] = self.root_dir+"/result_sp_data/"+self.data["xiushi"]+"/mesh/"
        self.data['data_render_folder'] = self.root_dir+"/result_sp_data/"+self.data["xiushi"]+"/render/"
        os.makedirs(self.data['data_point_folder'],exist_ok=True)
        os.makedirs(self.data['data_mesh_folder'],exist_ok=True)
        os.makedirs(self.data['data_render_folder'],exist_ok=True)
        self.copy_json_to_targets(self.data['data_point_folder'])
        self.copy_json_to_targets(self.data['data_mesh_folder'])
        self.copy_json_to_targets(self.data['data_render_folder'])
        # 样条数据存储位置
        self.data['spline_control_point_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/control_point/"
        self.data['spline_mesh_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/mesh/"
        self.data['spline_render_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/render/"
        self.data['spline_ray_tracing_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/ray_tracing/"

        os.makedirs(self.data['spline_control_point_folder'],exist_ok=True)
        os.makedirs(self.data['spline_mesh_folder'],exist_ok=True)
        os.makedirs(self.data['spline_render_folder'],exist_ok=True)
        os.makedirs(self.data['spline_ray_tracing_folder'],exist_ok=True)

        self.copy_json_to_targets(self.data['spline_control_point_folder'])
        self.copy_json_to_targets(self.data['spline_mesh_folder'])
        self.copy_json_to_targets(self.data['spline_render_folder'])
        self.copy_json_to_targets(self.data['spline_ray_tracing_folder'])

class sp_ot_JSONDataProcessor:
    def __init__(self, folder_path: str, encoding: str = 'utf-8'):
        self.root_dir = ROOT_DIR
        self.folder_path = os.path.join(self.root_dir, folder_path)
        self.encoding = encoding
        self.data = None
        self._load_data()
        self._path_set()
        # self.data_deal()

    def _load_data(self):
        try:
            with open(self.folder_path, 'r', encoding=self.encoding) as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"JSON文件 {self.folder_path} 不存在")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {str(e)}")
            
    def copy_json_to_targets(self, target_folder):
        os.makedirs(target_folder, exist_ok=True)
        original_name = os.path.basename(self.folder_path)
        json_target = os.path.join(target_folder, original_name)
        shutil.copy2(self.folder_path, json_target)

    def _path_set(self):
        # 输入图片
        self.data['image_path'] = self.root_dir+"/data/pic_data/{}.{}".format(self.data['picture_para']['name'],self.data['picture_para']['type'])
        self.data['image_txt'] = self.root_dir+"/data/pic_data/{}.{}".format(self.data['picture_para']['name'],"txt")
        # 所用修饰词
        self.data['xiushi'] = self.data['sample_point_para']['data_name']+"_"+self.data['test_num']

        # 样条数据存储位置
        self.data['spline_control_point_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/render_ot/control_point/"
        self.data['spline_render_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/render_ot/render/"
        self.data['spline_ray_tracing_folder'] = self.root_dir+"/result_spline/"+self.data["xiushi"]+"/ray_tracing_ot/"

        os.makedirs(self.data['spline_control_point_folder'],exist_ok=True)
        os.makedirs(self.data['spline_render_folder'],exist_ok=True)
        os.makedirs(self.data['spline_ray_tracing_folder'],exist_ok=True)

        self.copy_json_to_targets(self.data['spline_control_point_folder'])
        self.copy_json_to_targets(self.data['spline_render_folder'])
        self.copy_json_to_targets(self.data['spline_ray_tracing_folder'])
