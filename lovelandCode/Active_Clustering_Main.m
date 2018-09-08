load('mnist_all.mat');

% show example image ----------------------------------------------------

% get 1 image from each class to serve as example

% figure;

% for d = 0:9

%     % get a row

%     row(d+1,:) = 256 - uint8(eval(['train',int2str(d),'(1,:)']));

%     digit_im = reshape(row(d+1,:),28,28);

%     subplot(2,5,d+1);

%     imagesc(digit_im');

% %     colormap gray;

% end

% create combined data array ---------------------------------------------

% combine all training data into single vector for labels and 2-D array

% of 60,000x784 for image data

labels = [];

data = [];



for d = 0:9

    % get number of images for a given digit

    num_d = eval(['size(train',int2str(d),',1)']);

    clear these_labels;

    clear this_d_data;

    these_labels(1:num_d,1) = d;

    labels = [labels;these_labels];

    this_d_data(1:num_d,:) = eval(['train',int2str(d)]);

    data = [data;this_d_data];

    % note that e.g. num_digit_p1(1) has num of class 0 data vectors,etc.

    num_digit_p1(d+1) = numel(find(labels == d));

end

data = double(data);

% % show 3-D PCA data ----------------------------------------------------

% [coeff,score,latent] = pca(data,'NumComponents',3);

% colormap jet;

% c_map = colormap;

% figure; hold on;

% end_ind = 0;

% for d = 0:9

%     start_ind = end_ind + 1;

%     end_ind = start_ind + num_digit_p1(d+1) - 1;

%     color_triplet = c_map(d*6+1,:);

% %     plot3(score(start_ind:end_ind,1), ...

% %         score(start_ind:end_ind,2),score(start_ind:end_ind,3), ...

% %         '.','Color',color_triplet);

% %     plot(score(start_ind:end_ind,1), ...

% %         score(start_ind:end_ind,2), ...

% %         '.','Color',color_triplet);

% %     text(score(start_ind:10:end_ind,1), ...

% %         score(start_ind:10:end_ind,2), ...

% %         num2str(d),'Color',color_triplet);

%     text(score(start_ind:10:end_ind,1), ...

%         score(start_ind:10:end_ind,2), ...

%         score(start_ind:10:end_ind,3), ...

%         num2str(d),'Color',color_triplet);

% end

% axis([-1500 2500 -1500 1500 -1500 1500])

% grid on;

% legend('Class 0', ...

% 'Class 1', ...

% 'Class 2', ...

% 'Class 3', ...

% 'Class 4', ...

% 'Class 5', ...

% 'Class 6', ...

% 'Class 7', ...

% 'Class 8', ...

% 'Class 9');

% 

% title('3-D PCA of MNIST');

% select subset of data

dec_factor = input('Enter decimation factor: [1]');

if isempty(dec_factor)

  dec_factor = 1;

end

end_index = input('Enter ending index factor: [10000]');

if isempty(end_index)

  end_index = 10000;

end

mode = input('Enter mode: (0 for user labels, 1 for auto)');

if ~((mode ~=0) || (mode ~= 1))

    disp('Bad Mode - default is auto');

    mode = 1;

end



data_10 = data(1:dec_factor:end_index,:);

labels_10 = labels(1:dec_factor:end_index);



num_pending_clusters = 1;

pending_cluster_inds{1} = 1:size(data_10,1);

next_done_ind = 1;

done_labels = [];

done_cluster_inds = {};

digit_fig = figure;

r = 1;

while (num_pending_clusters > 0)

    % get next cluster label from pending list

    this_cluster_inds = pending_cluster_inds{1};

    data_this_cluster = data_10(this_cluster_inds,:);

    labels_this_cluster = labels_10(this_cluster_inds,:);

    % if cluster of 1

    if (size(data_this_cluster,1) == 1)

        if mode == 0 % ask for a label

            % show to user

            single_fig = figure;

            digit_im = reshape(data_this_cluster,28,28);

            imagesc(digit_im');

            axis equal; axis tight;

            label_digit = input('Enter a (numerical) label: ');

            close(single_fig);

        else

            label_digit = labels_this_cluster;

        end

        user_labels(r) = label_digit;

        % put cluster into done list

        done_labels(next_done_ind) = label_digit;

        done_cluster_inds{next_done_ind} = this_cluster_inds;

        next_done_ind = next_done_ind + 1;

        % remove current cluster from pending list

        pending_cluster_inds(1) = [];

        num_pending_clusters = num_pending_clusters - 1;

    % else - more than cluster of 1 so split into 2 clusters

    else

%         % Method 1

        [c, ctrs] = kmeans(data_this_cluster, 2);

        % find actual data points closest to centers

        ctrs1 = ctrs(1,:);

        ind = dsearchn(data_this_cluster,ctrs1);

        dpt_ctr1 = data_this_cluster(ind,:);

        label_ctr1 = labels_this_cluster(ind);

        ctrs2 = ctrs(2,:);

        ind = dsearchn(data_this_cluster,ctrs2);

        dpt_ctr2 = data_this_cluster(ind,:);

        label_ctr2 = labels_this_cluster(ind);



        ctrs = [dpt_ctr1;dpt_ctr2];

        

% % more below...

%         % Method 2

%         % find 2 points furthest from each other in cluster

% %         ok - that's expensive - how about

% %         find pt farthest from center

%         center_pt = mean(data_this_cluster);

%         d = pdist2(data_this_cluster,center_pt);

%         [~,far_pt1_ind] = max(d);

%         far_pt1 = data_this_cluster(far_pt1_ind,:);

% %         then find pt farthest from that pt

%         d = pdist2(data_this_cluster,far_pt1);

%         [~,far_pt2_ind] = max(d);

%         far_pt2 = data_this_cluster(far_pt2_ind,:);

% % more Method 1B...

% %         start_ctrs = [far_pt1;far_pt2];

% %         [c, ctrs] = kmeans(data_this_cluster, 2,'Start',start_ctrs);

% %         ctrs = start_ctrs;

% % Method 2

% % split cluster into halves based on these distances from the far points

%         d = pdist2(data_this_cluster,far_pt2);

%         max_dist = max(d);

%         c = zeros(length(data_this_cluster));

%         near_inds = find(d < max_dist/2)

%         far_inds = find(d >= max_dist/2)

%         c(near_inds) = 1;

%         c(far_inds) = 2;

%         ctrs = [far_pt1;far_pt2];

%  

        if mode == 0

            % show user centers 

            figure(digit_fig);

            for n = 1:2

                digit_im = reshape(ctrs(n,:),28,28);

                subplot(1,2,n);

                imagesc(digit_im');

                axis equal; axis tight;

            end

            title('Comparison Window');



            % check if same

            commandwindow;

           rl = input('Enter label if same - otherwise hit ''Enter'': ');

           if isempty(rl)

              rl = -1;

           end

        else

            if (label_ctr1 ~= label_ctr2)

                rl = -1;

            else

                rl = label_ctr1;

            end

        end

            

        

       reply(r) = rl;

       user_labels(r) = -1;



        if (reply(r) ~= -1) % if same

            % ask for a label

            label_digit = rl;

            user_labels(r) = label_digit;

            % put cluster into done list

            done_labels(next_done_ind) = label_digit;

            done_cluster_inds{next_done_ind} = this_cluster_inds;

            next_done_ind = next_done_ind + 1;

            % remove current cluster from pending list

            pending_cluster_inds(1) = [];

            num_pending_clusters = num_pending_clusters - 1;

        else % different labels

            % remove current cluster from pending list

            pending_cluster_inds(1) = [];

            num_pending_clusters = num_pending_clusters - 1;

            % add new sub-clusters to pending list

            for n = 1:2

                num_pending_clusters = num_pending_clusters + 1;

                this_cluster_inds_local = find(c == n);

                % NOTE: this changes inds back to being relative to original

                % data - NOT local subset

                pending_cluster_inds{num_pending_clusters} = this_cluster_inds(this_cluster_inds_local);

            end

        end

    end

    r = r + 1;

end



% recombine all clusters with same label in done list

final_cluster_labels_unique = [];

final_cluster_inds = {};

cluster_labels = zeros(size(labels_10));

j = 1;

for n = unique(done_labels)

    final_cluster_labels_unique = [final_cluster_labels_unique;n];  

    this_label_inds = find(done_labels == n);

    f = [];

    for k = this_label_inds

        f = [f;done_cluster_inds{k}'];

    end

    final_cluster_inds{j} = f;

    cluster_labels(final_cluster_inds{j}) = n;

    

    j = j + 1;

end



[dec_factor, end_index]

cf = confusionmat(labels_10,cluster_labels)

disp(['Accuracy: ',num2str(trace(cf)/sum(cf(:))*100)])

disp(['Number comparisons: ',num2str(length(reply))])

num_labels = numel(unique(user_labels))-1;

disp(['Number labels: ',num2str(num_labels)]);



close all;

file_name_string = ['out_',num2str(dec_factor),'_',num2str(end_index)];

save(file_name_string); 



% write code to evaluate test data

% find cluster centers of all clusters

num_clusters = length(done_labels);

for n = 1:num_clusters

    this_cluster_data = data_10(done_cluster_inds{n},:);

    centers(n,:) = mean(this_cluster_data);

end

% now find closest center for each point

% done_labels(1)

% IDX = knnsearch(centers,test1)

% numel(find(IDX ~= 1))

% done_labels(1)

% IDX = knnsearch(centers,test0)

% numel(find(IDX ~= 2))

















        

    



