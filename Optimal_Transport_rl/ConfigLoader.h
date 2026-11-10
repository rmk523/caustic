// JsonHandler.h
#ifndef JSON_HANDLER_H
#define JSON_HANDLER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class JsonHandler {
public:
    explicit JsonHandler(const std::string& filePath = "") {
        if (!filePath.empty()) {
            loadFromFile(filePath);
        }
    }
    
    bool loadFromFile(const std::string& filePath) {
        clearError();  // 清空之前的错误信息

        std::ifstream inputFile(filePath);
        if (!inputFile.is_open()) {
            setError("Failed to open JSON file: " + filePath);
            return false;
        }

        try {
            inputFile >> jsonData_;
            return true;  // 成功加载
        } 
        catch (const json::parse_error& e) {
            setError("JSON parse error: " + std::string(e.what()));
        }
        catch (...) {
            setError("Unknown error occurred while parsing JSON");
        }
        
        return false;  // 任何异常都会走到这里
    }

    template<typename T>
    T getValue(const std::string& key) const {
        if (!jsonData_.contains(key)) {
            throw std::runtime_error("Key does not exist: " + key);
        }
        try {
            return jsonData_[key].get<T>();
        } catch (const json::exception& e) {
            throw std::runtime_error("Type error for key '" + key + "': " + e.what());
        }
    }

    template<typename T>
    std::vector<T> getArray(const std::string& key) const {
        if (!jsonData_.contains(key) || !jsonData_[key].is_array()) {
            throw std::runtime_error("Invalid array key: " + key);
        }
        try {
            return jsonData_[key].get<std::vector<T>>();
        } catch (const json::exception& e) {
            throw std::runtime_error("Array type error for key '" + key + "': " + e.what());
        }
    }

    JsonHandler getObject(const std::string& key) const {
        if (!jsonData_.contains(key) || !jsonData_[key].is_object()) {
            throw std::runtime_error("Key '" + key + "' is not a JSON object");
        }
        JsonHandler handler;
        handler.jsonData_ = jsonData_[key]; // 或使用 std::move
        return handler;
    }

private:
    json jsonData_;

    std::string lastError_;
    void setError(const std::string& message) {
        lastError_ = message;
    }

    void clearError() {
        lastError_.clear();
    }
};

#endif // JSON_HANDLER_H