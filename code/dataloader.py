import torch
from torch.utils.data import Dataset
import pandas as pd
import parse
from parse import args
import scipy.sparse as sp
import torch.nn.functional as F
import networkx as nx
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class MyDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device
        # train dataset
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_pos_data = train_data[train_data[2] >= args.offset]
        train_neg_data = train_data[train_data[2] < args.offset]
        self.train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_pos_user = torch.from_numpy(train_pos_data[0].values).to(self.device)
        self.train_pos_item = torch.from_numpy(train_pos_data[1].values).to(self.device)
        self.train_pos_unique_users = torch.unique(self.train_pos_user)
        self.train_pos_unique_items = torch.unique(self.train_pos_item)
        self.train_neg_user = torch.from_numpy(train_neg_data[0].values).to(self.device)
        self.train_neg_item = torch.from_numpy(train_neg_data[1].values).to(self.device)
        self.train_neg_unique_users = torch.unique(self.train_neg_user)
        self.train_neg_unique_items = torch.unique(self.train_neg_item)
        # valid dataset
        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_pos_data = valid_data[valid_data[2] >= args.offset]
        valid_neg_data = valid_data[valid_data[2] < args.offset]
        self.valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_pos_user = torch.from_numpy(valid_pos_data[0].values).to(self.device)
        self.valid_pos_item = torch.from_numpy(valid_pos_data[1].values).to(self.device)
        self.valid_pos_unique_users = torch.unique(self.valid_pos_user)
        self.valid_pos_unique_items = torch.unique(self.valid_pos_item)
        self.valid_neg_user = torch.from_numpy(valid_neg_data[0].values).to(self.device)
        self.valid_neg_item = torch.from_numpy(valid_neg_data[1].values).to(self.device)
        self.valid_neg_unique_users = torch.unique(self.valid_neg_user)
        self.valid_neg_unique_items = torch.unique(self.valid_neg_item)
        # test dataset
        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_pos_data = test_data[test_data[2] >= args.offset]
        test_neg_data = test_data[test_data[2] < args.offset]
        self.test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_pos_user = torch.from_numpy(test_pos_data[0].values).to(self.device)
        self.test_pos_item = torch.from_numpy(test_pos_data[1].values).to(self.device)
        self.test_pos_unique_users = torch.unique(self.test_pos_user)
        self.test_pos_unique_items = torch.unique(self.test_pos_item)
        self.test_neg_user = torch.from_numpy(test_neg_data[0].values).to(self.device)
        self.test_neg_item = torch.from_numpy(test_neg_data[1].values).to(self.device)
        self.test_neg_unique_users = torch.unique(self.test_neg_user)
        self.test_neg_unique_items = torch.unique(self.test_neg_item)
        self.num_users = max([self.train_pos_unique_users.max(),
                              self.train_neg_unique_users.max(),
                              self.valid_pos_unique_users.max(),
                              self.valid_neg_unique_users.max(),
                              self.test_pos_unique_users.max(),
                              self.test_neg_unique_users.max()]).cpu()+1
        self.num_items = max([self.train_pos_unique_items.max(),
                              self.train_neg_unique_items.max(),
                              self.valid_pos_unique_items.max(),
                              self.valid_neg_unique_items.max(),
                              self.test_pos_unique_items.max(),
                              self.test_neg_unique_items.max()]).cpu()+1
        self.num_nodes = self.num_users+self.num_items
        print('users: %d, items: %d.' % (self.num_users, self.num_items))
        print('train: %d pos + %d neg.' % (self.train_pos_user.shape[0], self.train_neg_user.shape[0]))
        print('valid: %d pos + %d neg.' % (self.valid_pos_user.shape[0], self.valid_neg_user.shape[0]))
        print('test: %d pos + %d neg.' % (self.test_pos_user.shape[0], self.test_neg_user.shape[0]))
        
        self._train_neg_list = None
        self._train_pos_list = None
        self._valid_neg_list = None
        self._valid_pos_list = None
        self._test_neg_list = None
        self._test_pos_list = None
        self._A_pos = None
        self._A_neg = None
        self._degree_pos = None
        self._degree_neg = None
        self._tildeA = None
        self._tildeA_pos = None
        self._tildeA_neg = None
        self._indices = None
        self._paths = None
        self._values = None
        self._counts = None
        self._counts_sum = None
        self._L = None
        self._L_pos = None
        self._L_neg = None

        self._L_eigs = None
        self._L_eigs_high = None

        self._motif_adj = None
        self._motif_ids = None

    @ property
    def train_pos_list(self):
        if self._train_pos_list is None:
            self._train_pos_list = [list(self.train_pos_item[self.train_pos_user == u].cpu().numpy()) for u in range(self.num_users)]
        return self._train_pos_list

    @ property
    def train_neg_list(self):
        if self._train_neg_list is None:
            self._train_neg_list = [list(self.train_neg_item[self.train_neg_user == u].cpu().numpy()) for u in range(self.num_users)]
        return self._train_neg_list

    @ property
    def valid_pos_list(self):
        if self._valid_pos_list is None:
            self._valid_pos_list = [list(self.valid_pos_item[self.valid_pos_user == u].cpu().numpy()) for u in self.valid_pos_unique_users]
        return self._valid_pos_list

    @ property
    def valid_neg_list(self):
        if self._valid_neg_list is None:
            self._valid_neg_list = [list(self.valid_neg_item[self.valid_neg_user == u].cpu().numpy()) for u in self.valid_pos_unique_users]
        return self._valid_neg_list

    @ property
    def test_pos_list(self):
        if self._test_pos_list is None:
            self._test_pos_list = [list(self.test_pos_item[self.test_pos_user == u].cpu().numpy()) for u in self.test_pos_unique_users]
        return self._test_pos_list

    @ property
    def test_neg_list(self):
        if self._test_neg_list is None:
            self._test_neg_list = [list(self.test_neg_item[self.test_neg_user == u].cpu().numpy()) for u in self.test_pos_unique_users]
        return self._test_neg_list

    @ property
    def A_pos(self):
        if self._A_pos is None:
            self._A_pos = torch.sparse_coo_tensor(
                torch.cat([
                    torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                    torch.stack([self.train_pos_item+self.num_users, self.train_pos_user])], dim=1),
                torch.ones(self.train_pos_user.shape[0]*2).to(parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
        return self._A_pos

    @ property
    def degree_pos(self):
        if self._degree_pos is None:
            self._degree_pos = self.A_pos.sum(dim=1).to_dense()
        return self._degree_pos

    @ property
    def tildeA_pos(self):
        if self._tildeA_pos is None:
            D = self.degree_pos.float()
            D[D == 0.] = 1.
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._tildeA_pos = torch.sparse.mm(torch.sparse.mm(D1, self.A_pos), D2)
        return self._tildeA_pos

    @ property
    def L_pos(self):
        if self._L_pos is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._L_pos = D-self.tildeA_pos
        return self._L_pos

    @ property
    def A_neg(self):
        if self._A_neg is None:
            self._A_neg = torch.sparse_coo_tensor(
                torch.cat([
                    torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
                    torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1),
                torch.ones(self.train_neg_user.shape[0]*2).to(parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
        return self._A_neg

    @ property
    def degree_neg(self):
        if self._degree_neg is None:
            self._degree_neg = self.A_neg.sum(dim=1).to_dense()
        return self._degree_neg

    @ property
    def tildeA_neg(self):
        if self._tildeA_neg is None:
            D = self.degree_neg.float()
            D[D == 0.] = 1.
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._tildeA_neg = torch.sparse.mm(torch.sparse.mm(D1, self.A_neg), D2)
        return self._tildeA_neg

    @ property
    def L_neg(self):
        if self._L_neg is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._L_neg = D-self.tildeA_neg
        return self._L_neg

    @ property
    def L(self):
        if self._L is None:
            self._L = ((self.L_pos + args.alpha * self.L_neg))/(1+args.alpha)
            #self._L = (self.L_pos + self.L_neg)
        return self._L

    @property
    def L_eigs(self):

        if self._L_eigs is None:
            if args.eigs_dim == 0:
                self._L_eigs = torch.tensor([]).to(parse.device)
            else:
                _, self._L_eigs = sp.linalg.eigs(
                    sp.csr_matrix(
                        (self.L._values().cpu(), self.L._indices().cpu()),
                        (self.num_nodes, self.num_nodes)
                    
                    ),
                    k = args.eigs_dim,
                    which='SR'
                )
                self._L_eigs = torch.tensor(self._L_eigs.real).to(parse.device)

                # self._L_eigs = F.layer_norm(self._L_eigs, normalized_shape=(self._L_eigs.shape[-1],))

        return self._L_eigs

    @property
    def L_eigs_h(self):

        if self._L_eigs_high is None:
            if args.eigs_dim == 0:
                self._L_eigs_high = torch.tensor([]).to(parse.device)
            else:
                _, self._L_eigs_high = sp.linalg.eigs(
                    sp.csr_matrix(
                        (self.L._values().cpu(), self.L._indices().cpu()),
                        (self.num_nodes, self.num_nodes)
                    ),
                    k = args.eigs_dim,
                    which='LR'
                )
                self._L_eigs_high = torch.tensor(self._L_eigs_high.real).to(parse.device)

                # self._L_eigs_high = F.layer_norm(self._L_eigs_high, normalized_shape=(self._L_eigs_high.shape[-1],))

        return self._L_eigs_high

    @property
    def motif_adj(self):
        if self._motif_adj is not None:
            return self._motif_adj

        # 캐시 파일 경로 설정 (예: cache 폴더 내에 저장)
        cache_file = os.path.join(f"cache/{args.data}", f"motif_adj_kmeansknn_{args.data}_k{args.n_neighbors}_nClust{args.num_motifs}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached motif_adj from {cache_file}")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            indices = torch.tensor(cache_data["indices"], dtype=torch.long, device=self.device)
            values = torch.tensor(cache_data["values"], device=self.device)
            size = cache_data["size"]
            self._motif_adj = torch.sparse_coo_tensor(indices, values, size=size)
            self._motif_ids = torch.tensor(cache_data["motif_ids"], dtype=torch.long, device=self.device)
            return self._motif_adj

        # 1. 노드 임베딩 준비 (Laplacian의 높은 고유벡터 사용)
        eigs = self.L_eigs_h
        node_emb = eigs.cpu().numpy()

        # 2. KMeans 클러스터링으로 노드들을 그룹(모티프)으로 분할
        kmeans = KMeans(n_clusters=args.num_motifs, random_state=args.seed)
        labels = kmeans.fit_predict(node_emb)
        # 각 노드의 클러스터(모티프) ID를 저장
        self._motif_ids = torch.tensor(labels, dtype=torch.long, device=self.device)

        # 3. 각 클러스터 내부에서 k‑NN으로 연결 구축
        rows, cols = [], []
        # 클러스터별로 반복
        for cluster in range(args.num_motifs):
            # 해당 클러스터에 속한 노드 인덱스 추출 (numpy array)
            cluster_indices = np.where(labels == cluster)[0]
            if cluster_indices.size > 1:
                # 클러스터 내부의 임베딩 추출
                cluster_emb = node_emb[cluster_indices]
                # 클러스터의 크기에 따라 n_neighbors 설정 (자기 자신 포함하여 최소값)
                n_neighbors = min(args.n_neighbors, cluster_indices.size)
                knn_model = NearestNeighbors(n_neighbors=n_neighbors)
                knn_model.fit(cluster_emb)
                distances, knn_indices = knn_model.kneighbors(cluster_emb)
                # 자기 자신은 첫 번째 컬럼에 있으므로 제거
                knn_indices = knn_indices[:, 1:]
                # 각 노드에 대해 k‑NN 이웃 정보를 엣지 리스트에 추가
                for i in range(cluster_indices.size):
                    current_node = cluster_indices[i]
                    for neighbor_local_idx in knn_indices[i]:
                        neighbor_global_idx = cluster_indices[neighbor_local_idx]
                        rows.append(current_node)
                        cols.append(neighbor_global_idx)
            # 만약 클러스터에 단 한 개의 노드밖에 없다면 연결을 만들지 않습니다.
        
        # 4. 그래프의 대칭성을 위해, (i, j)와 (j, i)를 모두 추가
        rows_sym = list(cols)
        cols_sym = list(rows)
        rows.extend(rows_sym)
        cols.extend(cols_sym)

        # 5. 리스트를 tensor로 변환 및 값 설정
        rows = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols = torch.tensor(cols, dtype=torch.long, device=self.device)
        vals = torch.ones(rows.size(0), device=self.device)

        # 6. sparse COO tensor 생성
        self._motif_adj = torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            vals,
            size=(self.num_nodes, self.num_nodes)
        )

        # 7. 결과 캐시 파일에 저장
        os.makedirs(f"cache/{args.data}", exist_ok=True)
        with open(cache_file, "wb") as f:
            cache_data = {
                "indices": self._motif_adj._indices().cpu().numpy(),
                "values": self._motif_adj._values().cpu().numpy(),
                "size": self._motif_adj.size(),
                "motif_ids": self._motif_ids.cpu().numpy()
            }
            pickle.dump(cache_data, f)
            print(f"Saved motif_adj to cache file: {cache_file}")

        return self._motif_adj

    @property
    def motif_ids(self):
        if self._motif_ids is None:
            _ = self.motif_adj
        return self._motif_ids

    def sample(self):
        if self._indices is None:
            self._indices = torch.cat([
                torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                torch.stack([self.train_pos_item+self.num_users, self.train_pos_user]),
                torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
                torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1)
            self._paths = torch.cat([
                torch.ones(self.train_pos_user.shape).repeat(2),
                torch.zeros(self.train_neg_user.shape).repeat(2)], dim=0).long().to(parse.device)
            sorted_indices = torch.argsort(self._indices[0, :])
            self._indices = self._indices[:, sorted_indices]
            self._paths = self._paths[sorted_indices]
            self._counts = torch.bincount(self._indices[0], minlength=self.num_nodes)
            self._counts_sum = torch.cumsum(self._counts, dim=0)
            d = torch.sqrt(self._counts)
            d[d == 0.] = 1.
            d = 1./d
            self._values = torch.ones(self._indices.shape[1]).to(
                parse.device)*d[self._indices[0]]*d[self._indices[1]]
        res_X, res_Y = [], []
        record_X = []
        X,  Y,  = self._indices,  torch.ones_like(self._paths).long()*2+self._paths
        loop_indices = torch.zeros_like(Y).bool()
        for hop in range(args.sample_hop):
            loop_indices = loop_indices | (X[0] == X[1])
            for i in range(hop % 2, hop, 2):
                loop_indices = loop_indices | (record_X[i][1] == X[1])
            record_X.append(X)
            res_X.append(X[:, ~loop_indices])
            res_Y.append(Y[~loop_indices]-2)
            next_indices = self._counts_sum[X[1]]-(torch.rand(X.shape[1]).to(parse.device)*self._counts[X[1]]).long()-1
            X = torch.stack([X[0], self._indices[1, next_indices]], dim=0)
            Y = Y*2+self._paths[next_indices]
    
        # motif_ids는 미리 motif_adj property에서 계산해둔 self._motif_ids 사용
        motif_ids = self._motif_ids if self._motif_ids is not None else torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        
        return res_X, motif_ids

dataset = MyDataset(parse.train_file, parse.valid_file,
                    parse.test_file, parse.device)