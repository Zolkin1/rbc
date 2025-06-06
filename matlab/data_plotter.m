clc;
clear;
close all;

%%
steps = 10;
nq = 2;
nv = 2;
nu = 2;
dt = 0.02;

% TODO: Move to python

% TODO: Swap the obstacle with the h function
obstacle1_pos = [1, 0.5];
obstacle1_rad = 0.5;
obstacle2_pos = [0.3, 1.75];
obstacle2_rad = 0.35;

% Load the CSV
data = readmatrix('/home/zolkin/AmberLab/Project-Rollout-Certifications/mj_rollout/logs/log.csv');  % Each row = 1 trajectory rollout

num_rows = size(data, 1);
expected_cols = 1 + steps * (nq + nv + nu);

assert(size(data, 2) == expected_cols, 'Mismatch in expected CSV columns');

% Create time vector (step indices)
t = 0:steps-1;
t = t * dt;

q = [];
v = [];
u  = [];

% Loop over each row (i)
for row = 1:num_rows
    i_val = data(row, 1);

    offset = 2;  % Index after i
    
    q_flat = [];
    v_flat = [];
    u_flat = [];
    for j = 1:steps
        % Extract q (positions)
        q_flat = [q_flat, data(row, offset : offset + nq - 1)];
        offset = offset + nq;
    
        % Extract v (velocities)
        v_flat = [v_flat, data(row, offset : offset + nv - 1)];
        offset = offset + nv;
    end

    for j = 1:steps
        % Extract u (controls)
        u_flat = [u_flat, data(row, offset : offset + nu - 1)];
        offset = offset + nu;
    end

    q = [q; reshape(q_flat, [nq, steps])];
    v = [v; reshape(v_flat, [nv, steps])];
    u = [u; reshape(u_flat, [nu, steps])];
end

% Plot
figure;
hold on;
for i = 1:num_rows
    plot(q(2*(i-1) + 1,:), q(2*(i-1) + 2,:), 'LineWidth', 2)
    scatter(q(2*(i-1) + 1,1), q(2*(i-1) + 2,1), "filled")
end
hold off;
circle(obstacle1_pos(1), obstacle1_pos(2), obstacle1_rad);
circle(obstacle2_pos(1), obstacle2_pos(2), obstacle2_rad);
axis equal;
grid on;
title(sprintf('Row %d: x vs y', i_val));
xlabel('x'); ylabel('y');

% figure;
% subplot(2,1,1);
% plot(t, v');
% title('v vs time');
% xlabel('Time step'); ylabel('v');
% legend(arrayfun(@(i) sprintf('v%d', i), 1:nv, 'UniformOutput', false));
% 
% subplot(2,1,2);
% plot(t, u');
% title('u vs time');
% xlabel('Time step'); ylabel('u');
% legend(arrayfun(@(i) sprintf('u%d', i), 1:nu, 'UniformOutput', false));

%% Helper Functions
function h = circle(x,y,r)
    hold on
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit);
    hold off
end