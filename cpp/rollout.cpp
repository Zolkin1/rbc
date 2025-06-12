#include <chrono>
#include <iostream>
#include <fstream>
#include <array>

#include <yaml-cpp/yaml.h>
#include <mujoco/mujoco.h>

#include "poisson.h"


struct RolloutData
{
    std::vector<double> states;
    std::vector<double> inputs;
    // TODO: Add user data
};

void Rollout(const mjModel* model, mjData* data, int steps, RolloutData& rd) {
    int nstates = model->nq + model->nv;
    for (int i = 0; i < steps; i++) {
        mj_step(model, data);
        for (int j = 0; j < model->nq; j++) {
            rd.states[i*nstates + j] = data->qpos[j];
            // std::cout <<
        }
        for (int j = 0; j < model->nv; j++) {
            rd.states[i*nstates + model->nq + j] = data->qvel[j];
        }
        for (int j = 0; j < model->nu; j++) {
            rd.inputs[i*model->nu + j] = data->ctrl[j];
        }
        // TODO: Add user data
    }
}

void ParallelRollout(std::vector<const mjModel*>& models, std::vector<mjData*> data,  int steps, std::vector<RolloutData>& rd) {
    #pragma omp parallel for
    for (int i = 0; i < models.size(); i++) {
        Rollout(models[i], data[i], steps, rd[i]);
    }
}

void LogToCsv(std::vector<double>& states, std::vector<double>& inputs, int rollout_num, std::ofstream& file) {
    file << rollout_num << ",";     // Log the rollout number
    for (size_t i = 0; i < states.size(); ++i) {
        file << states[i];
        file << ",";  // Comma between values
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        file << inputs[i];
        if (i != inputs.size() - 1) {
            file << ",";  // Comma between values
        }
    }
    file << "\n";  // Newline at end
}

void CtrlCallback(const mjModel* model, mjData* data) {
    // Control callback for the rollouts
    std::array<double, 2> target_pos = {data->userdata[0], data->userdata[1]};
    double& kp = data->userdata[2];
    double& kd = data->userdata[3];

    // TODO: I can fake making a harder to traverse area by adjusting the controller to be worse in that area.

    data->ctrl[0] = kp*(target_pos[0] - data->qpos[0]) + kd*(0 - data->qvel[0]);
    data->ctrl[1] = kp*(target_pos[1] - data->qpos[1]) + kd*(0 - data->qvel[1]);
}

/**
 * @brief Scan through a grid in the scene to look for cells that are occupied using the ray caster.
 *
 * @param model mujoco model
 * @param data mujoco data
 * @return vector w/ 0 indicating unoccupied
 */
std::vector<float> OccupancyGrid(const mjModel* model, mjData* data, const mjtByte* geom_group,
    const std::array<double, 2>& center_pos, const std::array<double, 2>& grid_size, double grid_resolution) {

    std::array<double, 2> grid_cells = {grid_size[0]/grid_resolution + 1, grid_size[1]/grid_resolution + 1};
    std::array<double, 2> start_point = {center_pos[0] - grid_size[0]/2, center_pos[1] - grid_size[1]/2};

    std::vector<float> occupancy_grid(grid_cells[0] * grid_cells[1]);

    for (int i = 0; i < grid_cells[0]; i++) {
        for (int j = 0; j < grid_cells[1]; j++) {
            std::array<double, 2> pos = {
                start_point[0] + i * grid_resolution,
                start_point[1] + j * grid_resolution
            };


            std::array<double, 3> ray_origin = {
                pos[0],
                pos[1],
                10
            };

            const std::array<double, 3> direction = { 0.0, 0.0, -1.0 };
            int geom_id[1] = { -1 };

            // Perform ray cast
            double dist = mj_ray(model, data, ray_origin.data(),
                direction.data(), geom_group, 1, -1, geom_id);

            if (dist != -1) {
                occupancy_grid[j*grid_cells[0] + i] = 0;
                // occupancy_grid[i*grid_cells[1] + j] = 1;
            } else {
                occupancy_grid[j*grid_cells[1] + i] = 1;
            }
        }
    }

    return occupancy_grid;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Pass the model.xml followed by either [rollout | occupancy]!";
        return 1;
    }

    // Load model
    char error[1000] = "Could not load model";
    mjModel* model = mj_loadXML(argv[1], nullptr, error, 1000);
    if (!model) {
        std::cerr << error << "\n";
        return 1;
    }

    const std::string function = std::string(argv[2]);

    // Parameters
    const std::array<double, 2> center_pos = {0, 0};
    const std::array<double, 2> grid_size = {6, 6};
    const double grid_resolution = 0.05;
    int steps = 30;

    if (function == "rollout") {
        int ic_grid_x = 10;
        int ic_grid_y = 10;
        float ic_grid_resolution = 0.5;

        int num_rollouts = ic_grid_x * ic_grid_y;

        std::array<double, 2> target_pos = {0, 0};
        double kp = 1;
        double kd = 0.5;
        std::vector<const mjModel*> models;
        std::vector<mjData*> data;
        for (int i = 0; i < ic_grid_x; i++) {
            for (int j = 0; j < ic_grid_y; j++) {
                models.push_back(model);
                data.push_back(mj_makeData(model));

                // Set initial condition
                data.back()->qpos[0] = ic_grid_resolution*i - (ic_grid_resolution * ((ic_grid_x-1)/2.));
                data.back()->qpos[1] = ic_grid_resolution*j - (ic_grid_resolution * ((ic_grid_y-1)/2.));

                // Set the target
                data.back()->userdata[0] = target_pos[0];
                data.back()->userdata[1] = target_pos[1];

                // Set the gains
                data.back()->userdata[2] = kp;
                data.back()->userdata[3] = kd;
            }
        }

        std::cout << "Model loaded successfully.\n";

        std::vector<RolloutData> rollouts;
        rollouts.resize(num_rollouts);
        int nstates = model->nq + model->nv;
        for (int i = 0; i < num_rollouts; i++) {
            rollouts[i].states.resize(steps*nstates);
            rollouts[i].inputs.resize(steps*model->nu);
        }

        // Set the closed loop controller
        mjcb_control = CtrlCallback;

        auto start = std::chrono::high_resolution_clock::now();
        ParallelRollout(models, data, steps, rollouts);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

        // Log data to a csv
        const std::string filename = "/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/rollout_log.csv";
        std::ofstream file(filename);  // Overwrites file if it exists
        if (!file.is_open()) {
            std::cerr << "Error: could not open file " << filename << "\n";
            return 1;
        }

        for (int i = 0; i < num_rollouts; i++) {
            LogToCsv(rollouts[i].states, rollouts[i].inputs, i, file);
        }
        file.close();

        std::cout << "nq: " << model->nq << ", nv: " << model->nv << ", nu: " << model->nu << "\n";
        std::cout << "steps: " << steps << ", dt: " << model->opt.timestep << ", num rollouts: " << num_rollouts << "\n";
    } else if (function == "occupancy") {
        mjData* data = mj_makeData(model);
        mj_step(model, data);

        mjtByte geom_group[mjNGROUP] = {0, 0, 1, 0, 0, 0};
        std::vector<float> grid = OccupancyGrid(model, data, geom_group, center_pos, grid_size, grid_resolution);

        // Log data to a csv
        const std::string filename = "/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/ocg_log.csv";
        std::ofstream file(filename);  // Overwrites file if it exists
        if (!file.is_open()) {
            std::cerr << "Error: could not open file " << filename << "\n";
            return 1;
        }

        for (size_t i = 0; i < grid.size(); ++i) {
            file << grid[i];
            if (i != grid.size() - 1) {
                file << ",";  // Comma between values
            }
        }

        file.close();
        std::cout << "Occupancy grid logged." << std::endl;

        // Create the h
        std::vector<float> hgrid = std::vector<float>(grid.size(), 0);
        std::array<double, 2> grid_cells = {grid_size[0]/grid_resolution + 1, grid_size[1]/grid_resolution + 1};
        std::cout << "grid cells[0]: " << grid_cells[0] << ", grid cells[1]: " << grid_cells[1] << "\n";
        for(int n = 0; n < grid_cells[0]*grid_cells[1]; n++) hgrid[n] = 0;

        solve_poisson_safety_function(hgrid.data(), grid.data());
        // float h0 = get_h0(hgrid, rx, ry);

        std::cout << "Poission safety function created." << std::endl;

        // Log data to a csv
        const std::string filename_h = "/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/hsafe_log.csv";
        std::ofstream file_h(filename_h);  // Overwrites file if it exists
        if (!file_h.is_open()) {
            std::cerr << "Error: could not open file " << filename_h << "\n";
            return 1;
        }

        for (size_t i = 0; i < hgrid.size(); ++i) {
            file_h << hgrid[i];
            if (i != hgrid.size() - 1) {
                file_h << "\n";  // new line between values
            }
        }

        file_h.close();
        std::cout << "Safety function logged." << std::endl;

    } else {
        std::cerr << "Unknown function!\n";
    }

    // Write everything to a yaml
    YAML::Node config;
    config["ocg_res"] = grid_resolution;
    config["ocg_size_x"] = grid_size[0];
    config["ocg_size_y"] = grid_size[1];
    config["steps"] = steps;
    config["dt"] = model->opt.timestep;

    // Write to file
    std::ofstream fout("/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/cpp/logs/config.yaml");
    fout << config;
    return 0;
}
