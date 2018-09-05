import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix

mnist_all = sio.loadmat('mnist_all.mat') # Need to figure out where data set is coming from

labels = [] # Need to figure out sizes or if this is correct
data = [] # Need to figure out sizes or if this is correct

for d in range(9):
	num_d = eval(('np.shape(train',str(d), ')[0]'))
	
	these_labels[0:num_d,1] = d
	
	labels = (labels, these_labels)

	this_d_data[0:num_d, :] = eval(['train', str(d)])
	
	data = (data, this_d_data)
	
	num_digit_p1[d] = np.size( (labels == d).nonzero() )

data.astype(double)	
	
	
print('Enter decimation factor: [1]')
dec_fector = input()

if not dec_fector:
	dec_fector = 1
	
print('Enter ending index factor: [10000]')
end_index = input()

if not end_index:
	end_index = 10000;

print('Enter mode: (0 for user labels, 1 for auto)')
mode = input()

if !((mode !=0) || (mode != 1)):
	print('Bad Mode - default is auto')

    mode = 1

	
data_10 = data[0:dec_fector:end_index, :]
labels_10 = labels[0:dec_fector:end_index]

num_pending_clusters = 1

pending_cluster_inds = np.arange(0, np.shape(data_10)[0])

next_done_ind = 0

done_labels = []

done_cluster_inds = []

r = 0

while num_pending_clusters > 0:
	# get next cluster label from pending list
	this_cluster_inds = pending_cluster_inds
	
	data_this_cluster = data_10[this_cluster_inds,:]
	
	labels_this_cluster = labels_10[this_cluster_inds,:]
	
	# if cluster of 1
	if (np.shape(data_this_cluster)[0] == 1):
		
		if mode == 0: # ask for a label
			# show to user
			digit_im = np.reshape(data_this_cluster, (28,28))
			
			plt.imshow(np.transpose(digit_im))
			plt.show()
			
			print('Enter a (numerical) label: ')
			label_digit = input()
			
			plt.close()
			
		else:
			label_digit = labels_this_cluster;
			
		user_labels[r] = label_digit;
		
		# put cluster into done list
		done_labels[next_done_ind] = label_digit
		
		done_cluster_inds[next_done_ind] = this_cluster_inds
		
		next_done_ind += 1
		
		# remove current cluster from pending list
		pending_cluster_inds[0] = []

		num_pending_clusters = num_pending_clusters - 1
		
	# else - more than cluster of 1 so split into 2 clusters	
	else:
		# Method 1
		#kmeans = KMeans(n_clusters=2).fit(X)
		#ctrs = kmeans.cluster_centers_
		ctrs = kmeans(data,2)
		c = vq(data,centroids)
		
		# find actual data points closest to centers
		ctrs1 = ctrs[0,:]
		
		tree = cKDTree(data_this_cluster)
		ind = tree.query(ctrs1)
		
		dpt_ctr1 = data_this_cluster[ind,:]
		
		label_ctr1 = labels_this_cluster[ind]
		
		ctrs2 = ctrs2[1,:]
		
		ind = tree.query(ctrs2)
		
		dpt_ctr2 = data_this_cluster[ind,:]
		
		label_ctr2 = labels_this_cluster[ind]
		
		ctrs = (dpt_ctr1,dpt_ctr2)
		
		if mode == 0:
			for n in range(1,3):
				digit_im = np.reshape(ctrs[n,:], (28, 28))
				
				plt.subplot(1,2,n)
				plt.imshow(np.transpose(digit_im))
				
			plt.title('Comparison WIndow')
			
			plt.show()
			
			# check if same
			print('Enter label if same - otherwise hit ''Enter'': ')
			r1 = input()
			
			if not r1:
				r1 = -1;
		
		else:
			if label_ctr1 != label_ctr2:
				r1 = -1
				
			else:
				r1 = label_ctr1
				
				
		reply[r] = r1
		
		user_labels[r] = -1
		
		
		if (reply[r] != -1): #if same
			# ask for label 
			label_digit = r1
			
			user_labels[r] = label_digit
			
			# put cluster into done list
			done_labels[next_done_ind] = label_digit

            done_cluster_inds[next_done_ind] = this_cluster_inds

            next_done_ind += 1;
			
			# remove current cluster from pending list

            pending_cluster_inds[0] = []

            num_pending_clusters -= 1
		
		else: # different labels
			# remove current cluster from pending list

            pending_cluster_inds[0] = []

            num_pending_clusters -= 1
			
			# add new sub-clusters to pending list

            for n in range(0,2):

                num_pending_clusters += 1

                this_cluster_inds_local = (c == n).nonzero()

                # NOTE: this changes inds back to being relative to original

                # data - NOT local subset

                pending_cluster_inds[num_pending_clusters] = this_cluster_inds[this_cluster_inds_local]
	
	r += 1

	
# recombine all clusters with same label in done list
final_cluster_labels_unique = []

final_cluster_inds = []

cluster_labels = np.zeros(np.shape(labels_10))

j = 0

for n in np.unique(done_labels):
	final_cluster_labels_unique = [final_cluster_labels_unique, n]
	
	this_labels_inds = (done_labels == n).nonzero()
	
	f = []
	
	for k in this_labels_inds:
		f = [f, np.transpose(done_cluster_inds[k])]
		
	final_cluster_inds[j] = f
	
	cluster_labels[final_cluster_inds[j]] = n
	
	j += 1
	
print(dec_fector) 
print(end_index)

cf = confusion_matrix(labels_10, cluster_labels)

print('Accuracy: ')
print(str(np.trace(cf)/np.sum(cf[:])*100))

print('Number comparisons: ')
print(str(np.size(reply)))

num_labels = np.size(np.unique(user_labels)) - 1
print('Number labels: ')
print(str(num_labels))



# Save all variables somehow



# write code to evaluate test data

# find cluster centers of all clusters

num_clusters = np.size(done_labels)

for n in range(0, num_clusters):

    this_cluster_data = data_10[done_cluster_inds[n],:]

    centers[n,:] = np.mean(this_cluster_data);

