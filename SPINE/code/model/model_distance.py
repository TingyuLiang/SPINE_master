import torch
from torch import nn
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.INFO)


class SPINEModel(torch.nn.Module):

	def __init__(self, params):
		super(SPINEModel, self).__init__()
		
		# params
		self.inp_dim = params['inp_dim']
		self.hdim = params['hdim']
		self.noise_level = params['noise_level']
		self.getReconstructionLoss = nn.MSELoss()
		self.rho_star = 1.0 - params['sparsity']
		self.k = params['k']		#!!!!!!!!!!!!!!!
		
		# autoencoder
		logging.info("Building model ")
		self.linear1 = nn.Linear(self.inp_dim, self.hdim)
		self.linear2 = nn.Linear(self.hdim, self.inp_dim)
		

	def forward(self, batch_x, batch_y):
		
		# forward
		batch_size = batch_x.data.shape[0]
		linear1_out = self.linear1(batch_x)
		h = linear1_out.clamp(min=0, max=1) # capped relu
		out = self.linear2(h)
		k = self.k

		# different terms of the loss
		reconstruction_loss = self.getReconstructionLoss(out, batch_y) # reconstruction loss
		psl_loss = self._getPSLLoss(h, batch_size) 		# partial sparsity loss
		asl_loss = self._getASLLoss(h)    	# average sparsity loss
		local_loss = self._get_Local_Loss(batch_y, h, batch_size, k)	# calculate the local loss
		#print(local_loss)
		total_loss = reconstruction_loss + psl_loss + asl_loss + local_loss
		
		return out, h, total_loss, [reconstruction_loss, psl_loss, asl_loss, local_loss]


	def _getPSLLoss(self,h, batch_size):
		return torch.sum(h*(1-h))/ (batch_size * self.hdim)


	def _getASLLoss(self, h):
		temp = torch.mean(h, dim=0) - self.rho_star
		temp = temp.clamp(min=0)
		return torch.sum(temp * temp) / self.hdim


	def _get_Local_Loss(self, batch_y, h, batch_size, k):

		#cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		dist = nn.PairwiseDistance(p=2)

		###### calculate distance between words for imput space
		Matrix_imput = torch.empty(batch_size, batch_size, dtype=torch.float)

		for row_imput_1 in range(0, batch_size):
			for column_imput_1 in range(row_imput_1, batch_size):

				if column_imput_1 == row_imput_1:
					Matrix_imput.data[row_imput_1][column_imput_1] = float("Inf") # not calculate distance for self
				else:
					index_row_imput_1 = torch.tensor([row_imput_1])
					index_column_imput_1 = torch.tensor([column_imput_1])
					batch_y_row_1 = torch.index_select(batch_y, 0, index_row_imput_1)
					batch_y_column_1 = torch.index_select(batch_y, 0, index_column_imput_1)

					Matrix_imput.data[row_imput_1][column_imput_1] = dist(batch_y_row_1, batch_y_column_1)

		for row_imput_2 in range(0, batch_size):
			for column_imput_2 in range(0, row_imput_2):
				Matrix_imput.data[row_imput_2][column_imput_2] = Matrix_imput.data[column_imput_2][row_imput_2]

		Matrix_imput_top_k = torch.topk(Matrix_imput,k,dim=1,largest=False, sorted=True)	# choose the k smallest values of each row
		Matrix_imput_top_k_sum = torch.sum(Matrix_imput_top_k[0], dim=1)	# sum theses k smallest values


		###### cacluate distance between words for embedding space
		Matrix_embedding = torch.empty(batch_size, batch_size, dtype=torch.float)

		for row_1 in range(0, batch_size):
			for column_1 in range(row_1, batch_size):

				if column_1 == row_1:
					Matrix_embedding.data[row_1][column_1] = float("Inf")
				else:
					index_row_embedding_1 = torch.tensor([row_1])
					index_column_embedding_1 = torch.tensor([column_1])
					h_row_1 = torch.index_select(h, 0, index_row_embedding_1)
					h_column_1 = torch.index_select(h, 0, index_column_embedding_1)

					Matrix_embedding.data[row_1][column_1] = dist(h_row_1, h_column_1)

		for row_2 in range(0, batch_size):
			for column_2 in range(0, row_2):
				Matrix_embedding.data[row_2][column_2] = Matrix_embedding.data[column_2][row_2]

		Matrix_embedding_top_k = torch.topk(Matrix_embedding,k,dim=1,largest=False, sorted=True)
		Matrix_embedding_top_k_sum = torch.sum(Matrix_embedding_top_k[0], dim=1)
		

		Mat_loss = torch.abs(torch.sub(Matrix_imput_top_k_sum[0], Matrix_embedding_top_k_sum[0]))

		local_loss_result = torch.mean(Mat_loss)

		return local_loss_result