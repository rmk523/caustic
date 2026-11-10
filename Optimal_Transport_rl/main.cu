#include <filesystem>
#include "ot_op.cuh"
#include "ConfigLoader.h"
using namespace std;
namespace fs = std::filesystem;

void copyJsonFile(const std::string& sourcePath, const std::string& destinationPath) {
    try {
        if (!fs::exists(sourcePath)) {
            std::cerr << "源文件不存在: " << sourcePath << std::endl;
            return;
        }
        fs::path destDir(destinationPath);
        if (!fs::exists(destDir)) {
            fs::create_directories(destDir);
        }
        fs::path sourceFile(sourcePath);
        fs::path destFile = destDir / sourceFile.filename();

        // 复制文件
        fs::copy_file(sourcePath, destFile, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

bool copyFileToFolder(const std::string& srcFile,
                      const std::string& dstFolder)
{
    try
    {
        fs::create_directories(dstFolder);
        fs::path srcPath(srcFile);
        fs::path dstPath = fs::path(dstFolder) / srcPath.filename();
        fs::copy_file(
            srcPath,
            dstPath,
            fs::copy_options::overwrite_existing
        );
        return true;
    }
    catch (const fs::filesystem_error& e)
    {
        std::cerr << "Copy failed: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Need one input number\n";
        return 1;
    }
    std::string ot_time = argv[1];
    std::string xiushi = argv[2];

    JsonHandler loader;
    std::string test_num_str;
    std::string data_name;

    std::string picture_name;
    std::string picture_path;
    int* pic_size;
    ScalarType* picture;
    
    int total_iter;
    ScalarType convergence_tolerance;
    int history_size,aa_history_size;
    ScalarType initial_step;
    ScalarType aa_fixed_para;
    bool if_aa_op_initial;

    ScalarType z_of_len;
    ScalarType z_of_receiver;
    ScalarType refraction_index;
    std::string light_type;
    ScalarType* point_light_coordinate = new ScalarType[3];
    ScalarType sample_domain[4];
    ScalarType proj_domain[4];
    ScalarType proj_corner[4];//xmin,xmax,ymin,ymax

    std::string points_path;
    std::string proj_xy_path;
    std::string left_idx_path,right_idx_path,top_idx_path,bottom_idx_path;

    if (loader.loadFromFile("../set_json/"+xiushi+"/_py_sp_settings.json"))
    {
        test_num_str = loader.getValue<std::string>("test_num");
        data_name = loader.getObject("sample_point_para").getValue<std::string>("data_name");

        picture_name = loader.getObject("picture_para").getValue<std::string>("name");
        picture_path = "../../data/pic_data/"+picture_name+".txt";

        total_iter = loader.getObject("sample_point_para").getValue<int>("total_iter");
        convergence_tolerance = loader.getObject("sample_point_para").getValue<ScalarType>("convergence_tolerance");

        history_size = loader.getObject("sample_point_para").getValue<int>("history_size");
        aa_history_size = loader.getObject("sample_point_para").getValue<int>("aa_history_size");
        aa_fixed_para = loader.getObject("sample_point_para").getValue<ScalarType>("aa_fixed_para");
        if_aa_op_initial = loader.getObject("sample_point_para").getValue<bool>("if_aa_op_initial");
        initial_step = loader.getObject("sample_point_para").getValue<ScalarType>("initial_step");

        z_of_len = loader.getObject("len_para").getValue<ScalarType>("z_of_len");
        z_of_receiver = loader.getObject("len_para").getValue<ScalarType>("z_of_receiver");
        refraction_index = loader.getObject("len_para").getValue<ScalarType>("refraction_index");
        light_type = loader.getObject("light_para").getValue<std::string>("light_type");
        std::vector<ScalarType> point_light_vector = loader.getObject("light_para").getArray<ScalarType>("point_light_coordinate");
        std::vector<ScalarType> sample_domain_vector = loader.getObject("sample_point_para").getArray<ScalarType>("sample_domain");
        std::vector<ScalarType> proj_domain_vector = loader.getObject("len_para").getArray<ScalarType>("proj_domain");
        for(int i=0; i<3; i++)
        {
            point_light_coordinate[i] = point_light_vector[i];
        }
        for(int i=0;i<4;i++)
        {
            sample_domain[i] = sample_domain_vector[i];
            proj_domain[i] = proj_domain_vector[i];
        }

        points_path = "../seed_point/" + xiushi + "/points.txt";
        proj_xy_path = "../seed_point/" + xiushi + "/proj_xy.txt";
        left_idx_path = "../seed_point/" + xiushi + "/left_idx.txt";
        right_idx_path = "../seed_point/" + xiushi + "/right_idx.txt";
        top_idx_path = "../seed_point/" + xiushi + "/top_idx.txt";
        bottom_idx_path = "../seed_point/" + xiushi + "/bottom_idx.txt";
                
        read_pic_txt_to_array( picture_path, picture, pic_size);
    }
    else
    {
        cout<<"cpp json文件读取失败！";
        return -1;
    }

    Eigen::VectorXd points;
    Eigen::VectorXd proj_xy;
    Eigen::VectorXi left_idx,right_idx,bottom_idx,top_idx;
    readPointsFlat(points_path,points);
    readPointsFlat(proj_xy_path,proj_xy);
    readIntVector(left_idx_path,left_idx);
    readIntVector(right_idx_path,right_idx);
    readIntVector(bottom_idx_path,bottom_idx);
    readIntVector(top_idx_path,top_idx);
    std::vector<Eigen::VectorXi> idx_list = {left_idx,right_idx,bottom_idx,top_idx};

    ScalarType x_min = std::numeric_limits<ScalarType>::max();
    ScalarType x_max = std::numeric_limits<ScalarType>::lowest();
    ScalarType y_min = std::numeric_limits<ScalarType>::max();
    ScalarType y_max = std::numeric_limits<ScalarType>::lowest();
    for (int i = 0; i < proj_xy.size(); i += 2)
    {
        ScalarType x = proj_xy(i);
        ScalarType y = proj_xy(i + 1);

        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
    }
    ScalarType x_scalar = (sample_domain[1] - sample_domain[0]),y_scalar = (sample_domain[3] - sample_domain[2]);

    proj_corner[0] = std::min(x_min, proj_domain[0]);
    if(proj_corner[0] == x_min)
    {
        proj_corner[0] = proj_corner[0] - 1;
    }

    proj_corner[1] = std::max(x_max, proj_domain[1]);
    if(proj_corner[1] == x_max)
    {
        proj_corner[1] = proj_corner[1] + 1;
    }

    proj_corner[2] = std::min(y_min, proj_domain[2]);
    if(proj_corner[2] == y_min)
    {
        proj_corner[2] = proj_corner[2] - 1;
    }
    
    proj_corner[3] = std::max(y_max, proj_domain[3]);
    if(proj_corner[3] == y_max)
    {
        proj_corner[3] = proj_corner[3] + 1;
    }
    
    const int points_num = points.size()/2;
    Eigen::VectorXd originPointWeights = Eigen::VectorXd::Zero(points_num);
    Eigen::VectorXd projPointWeights = Eigen::VectorXd::Zero(points_num);
    
    if(ot_time != "0")
    {   
        int prev_time = std::stoi(ot_time) - 1;
        std::string projPointWeightsFilename = "../../data/weights/" + xiushi + "/" + std::to_string(prev_time) + "/final_proj_point_weights.txt";
        read_weights_from_txt( projPointWeightsFilename, projPointWeights);
    }

    Eigen::VectorXd output_points(points_num*2);
    bool* if_exist = new bool[points_num];
    int N_threads=16;

    std::string inside_light_path = "../seed_point/" + xiushi + "/inside_light_direct.txt";
    std::string height_path = "../seed_point/" + xiushi + "/" + ot_time + "/height.txt";
    Eigen::VectorXd input_light(3*points_num);
    Eigen::VectorXd input_point_centers(2*points_num);
    Eigen::VectorXd inside_light(3*points_num);
    Eigen::VectorXd height(points_num);

    if (light_type == "parallel_light") {
        for (int i = 0; i < points_num; ++i) {
            input_light(3*i + 0) = 0.0;
            input_light(3*i + 1) = 0.0;
            input_light(3*i + 2) = 1.0;
        }
    } 
    else if (light_type == "point_light") {
        Eigen::Vector3d point_light(point_light_coordinate[0], point_light_coordinate[1], point_light_coordinate[2]);
        for (int i = 0; i < points_num; ++i) {
            double px = points(2*i);
            double py = points(2*i + 1);
            double pz = z_of_len;
            Eigen::Vector3d point_pos(px, py, pz);
            Eigen::Vector3d dir = point_pos - point_light;  // 方向从光源指向点
            dir.normalize();
            input_light(3*i + 0) = dir.x();
            input_light(3*i + 1) = dir.y();
            input_light(3*i + 2) = dir.z();
        }
    }
    else {
        std::cerr << "Unknown light_type: " << light_type << std::endl;
    }
    
    readPointsFlat(points_path,input_point_centers);

    std::ifstream inside_light_file(inside_light_path);
    if (!inside_light_file.is_open()) {
        std::cerr << "Failed to open file: " << inside_light_path << std::endl;
        return;
    }
    for (int i = 0; i < points_num; ++i) {
        double dx, dy, dz;
        inside_light_file >> dx >> dy >> dz;
        if (inside_light_file.fail()) {
            std::cerr << "Error reading direction for point " << i << std::endl;
            break;
        }
        Eigen::Vector3d dir(dx, dy, dz);
        dir.normalize();
        inside_light(3*i + 0) = dir.x();
        inside_light(3*i + 1) = dir.y();
        inside_light(3*i + 2) = dir.z();
    }
    inside_light_file.close();
    
    std::ifstream heightfile(height_path);
    if (!heightfile.is_open()) {
        std::cerr << "Failed to open file: " << height_path << std::endl;
        return;
    }
    for (int i = 0; i < points_num; ++i) {
        double dx;
        heightfile >> dx;
        if (heightfile.fail()) {
            std::cerr << "Error reading direction for point " << i << std::endl;
            break;
        }
        height(i) = dx;
    }
    heightfile.close();

    if(ot_time == "0")
    {
        direct_ot( points, originPointWeights, proj_xy, projPointWeights, picture, pic_size,
                    total_iter, convergence_tolerance, initial_step, history_size,
                    sample_domain, z_of_len, z_of_receiver, input_light,
                    light_type, point_light_coordinate, proj_domain, proj_corner,
                    idx_list, N_threads, xiushi, ot_time, output_points, if_exist); 
    }
    else
    {
        aarl_direct_ot( points, originPointWeights, proj_xy, projPointWeights, picture, pic_size,
                    total_iter, convergence_tolerance, initial_step, history_size, aa_history_size, aa_fixed_para,if_aa_op_initial,
                    sample_domain, z_of_len, z_of_receiver, input_light, input_point_centers, inside_light, height,
                    refraction_index, light_type, point_light_coordinate, proj_domain, proj_corner,
                    idx_list, N_threads, xiushi, ot_time, output_points, if_exist);
    } 

    std::string finalProjPointWeightsFileName = "../../data/weights/" + xiushi + "/" + ot_time + "/final_proj_point_weights.txt";
    std::filesystem::path finalProjPointWeightsFilePath(finalProjPointWeightsFileName);
    std::filesystem::create_directories(finalProjPointWeightsFilePath.parent_path());

    ofstream finalProjPointWeightsFile(finalProjPointWeightsFileName);
    for(int i=0;i<points_num;i++)
    {
        finalProjPointWeightsFile<<projPointWeights[i]<<std::endl;
    }
    finalProjPointWeightsFile.close();

    //输出点位置
    std::string out_center_txt ="../../data/point_data/" + xiushi + "/" + ot_time + "/out.txt";

    std::filesystem::path output_folder = "../../data/point_data/" + xiushi + "/" + ot_time;
    std::filesystem::create_directories(output_folder);

    std::string src = "../set_json/" + xiushi + "/_py_sp_settings.json";
    std::string dstFolder = "../../data/point_data/"+xiushi + "/"  + ot_time + "/";
    copyFileToFolder(src, dstFolder);

    if(ot_time == "0")
    {
        std::string in_center_txt = "../../data/point_data/" + xiushi + "/in.txt";
        ofstream in_center_file(in_center_txt);
        for(int i=0;i<points_num;i++)
        {
            if(if_exist[i] == true)
            {
                in_center_file<<points[2*i]<<" "<<points[2*i+1]<<std::endl;
            }
        }
        in_center_file.close();
    }

    ofstream out_center_file(out_center_txt);
    for(int i=0;i<points_num;i++)
    {
        if(if_exist[i] == true)
        {
            out_center_file<<output_points[2*i]<<" "<<output_points[2*i+1]<<std::endl;
        }
    }
    out_center_file.close();

    delete[] pic_size;
    delete[] picture;
    delete[] point_light_coordinate;
    delete[] if_exist;

    return 0;
}
