import world
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import odeblock as ode

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError
class LTOCF(BasicModel):
    """
    # LT-OCF model in PyTorch
    """
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LTOCF, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.__init_ode()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
    
    def __init_ode(self):
        """
        ## Init ODE blocks  and times

        - the number of time to be splited
        - ODE Block (learnable time or fixed time)

        """
        self.time_split = self.config['time_split'] 
        if world.config['learnable_time'] == True:
            
            self.odetimes = ode.ODETimeSetter(self.time_split, self.config['K'])
            self.odetime_1 = [self.odetimes[0]]
            self.odetime_2 = [self.odetimes[1]]
            self.odetime_3 = [self.odetimes[2]]
            self.ode_block_test_1 = ode.ODEBlockTimeFirst(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
            self.ode_block_test_2 = ode.ODEBlockTimeMiddle(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
            self.ode_block_test_3 = ode.ODEBlockTimeMiddle(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
            self.ode_block_test_4 = ode.ODEBlockTimeLast(ode.ODEFunction(self.Graph),self.time_split, self.config['solver'])
        else:
            self.odetime_splitted = ode.ODETimeSplitter(self.time_split, self.config['K'])
            self.ode_block_1 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], 0, self.odetime_splitted[0])
            self.ode_block_2 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], self.odetime_splitted[0], self.odetime_splitted[1])
            self.ode_block_3 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], self.odetime_splitted[1], self.odetime_splitted[2])
            self.ode_block_4 = ode.ODEBlock(ode.ODEFunction(self.Graph), self.config['solver'], self.odetime_splitted[2], self.config['K'])

    def get_time(self):
        ode_times=list(self.odetime_1)+ list(self.odetime_2)+ list(self.odetime_3)
        return ode_times
    
    def computer(self):
        """
        ## propagate methods for LT-OCF
        
        - $u$:
        - $p$: 
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        """
        ## Dual co-evolving ODEs
        \begin{align}
        \boldsymbol{u}(t_1) =&\; {\color{red}\boldsymbol{u}(0) +} \int_{0}^{t_1}f(\boldsymbol{p}(t))dt,\\
        \boldsymbol{p}(t_1) =&\; {\color{red}\boldsymbol{p}(0) +} \int_{0}^{t_1}g(\boldsymbol{u}(t))dt,\\
        \vdots\\
        \boldsymbol{u}(K) =&\; {\color{red}\boldsymbol{u}(t_T) +} \int_{t_T}^{K}f(\boldsymbol{p}(t))dt,\\
        \boldsymbol{p}(K) =&\; {\color{red}\boldsymbol{p}(t_T) +} \int_{t_T}^{K}g(\boldsymbol{u}(t))dt,
        \end{align}
        """
        if world.config['learnable_time'] == True:
            out_1 = self.ode_block_test_1(all_emb, self.odetime_1)
            if world.config['dual_res'] == False:
                out_1 = out_1 - all_emb
            embs.append(out_1)

            out_2 = self.ode_block_test_2(out_1, self.odetime_1, self.odetime_2)
            if world.config['dual_res'] == False:
                out_2 = out_2 - out_1
            embs.append(out_2)

            out_3 = self.ode_block_test_3(out_2, self.odetime_2, self.odetime_3)
            if world.config['dual_res'] == False:
                out_3 = out_3 - out_2
            embs.append(out_3)            

            out_4 = self.ode_block_test_4(out_3, self.odetime_3)
            if world.config['dual_res'] == False:
                out_4 = out_4 - out_3
            embs.append(out_4)
            
        elif world.config['learnable_time'] == False:
            all_emb_1 = self.ode_block_1(all_emb)
            all_emb_1 = all_emb_1 - all_emb
            embs.append(all_emb_1)
            all_emb_2 = self.ode_block_2(all_emb_1)
            all_emb_2 = all_emb_2 - all_emb_1
            embs.append(all_emb_2)
            all_emb_3 = self.ode_block_3(all_emb_2)
            all_emb_3 = all_emb_3 - all_emb_2
            embs.append(all_emb_3)
            all_emb_4 = self.ode_block_4(all_emb_3)
            all_emb_4 = all_emb_4 - all_emb_3
            embs.append(all_emb_4)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)        

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma