addpath('probe_data_map_matching');
data = 'probe_data_map_matching';
M = csvread(strcat(data,'\Link.csv'));
% ids = [762466209, 539290318, 565394943, 756999806, 536889642, 540611218, 572226303, 548057651];
ids = csvread('file.csv');
% uni = unique(M(:,1));
% ids = double(ids);
% ids(2)
hold on
for i=1:size(ids,1)
    x = find(M(:,1) == ids(i));
    plot(M(x,2),M(x,3),'-rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',5);
%     hold on;
end
hold off;
title('Link Plot');
xlabel('Latitude');
ylabel('Longitude');
% axis([51.17180, 51.191723, 10.95666, 10.99938]);
