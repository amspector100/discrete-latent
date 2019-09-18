import torch

def get_distance(a, b):
    """
    Computes pairwise distances between two sets of vectors
    param a: n x d dimension tensor
    param b: m x d dimension tensor
    returns: m x n dimension tensor of distances 
    """
    d = a.size(-1)
    a2 = a.pow(2).sum(-1).view(1, -1)
    b2 = b.pow(2).sum(-1).view(-1, 1)
    ab = torch.tensordot(b.view(-1, 1, d), a.view(1, -1, d), ([-1], [-1])) \
              .squeeze()
    distance = a2 + b2 - 2 * ab
    return distance

@torch.no_grad()
def get_distances(data, centers, device='cuda'):
    """ 
    Finds the nearest 'center' for each point in the data.
    param data: N x d dimension tensor of data vectors 
    (e.g. encoder outputs)
    param centers: K x d dimension tensor of VQ-VAE embedding vectors
    returns: (distances, indices)
    """
    #print("getting distances")
    d = data.size(-1)
    distances = []
    indices = []
    #print(data.device)
    for batch in data.split(2**14):
        torch.cuda.empty_cache()
        batch = batch.to(device=device)
        batch_distances, i = get_distance(batch, centers).min(0)
        distances += [batch_distances]
        indices += [i]
    return torch.cat(distances), torch.cat(indices)

@torch.no_grad()
def lloyd(data, centers, weights=None, device='cuda'):
    """
    Runs lloyd kmeans given initial centers for the data
    param data: N x d dimension tensor of data vectors 
    (e.g. encoder outputs)
    param centers: K x d dimension tensor of VQ-VAE embedding vectors
    returns: optimal new centers
    """
    n, d = data.size()
    k = len(centers)
    old_centers = torch.zeros_like(centers)
    
    while (centers - old_centers).pow(2).sum(-1).max() > 0.3:
        
        old_centers = centers.clone()
        _, indices = get_distances(data, centers, device=device)
        for i in range(k):
            cluster_indices = (indices == i).nonzero().view(-1) \
                                .to(device=data.device)
            if len(cluster_indices) == 0:
                continue
            cluster_points = data.gather(0, cluster_indices
                                            .view(-1, 1).expand(-1, d)) \
                                 .to(device=device)
            if weights is not None:
                cluster_weights = weights.gather(0, cluster_indices).view(-1, 1)
                centers[i] = (cluster_points * cluster_weights).sum(0) / \
                              cluster_weights.sum()
            else:
                centers[i] = cluster_points.mean(0)
        #print((centers - old_centers).pow(2).sum(-1).max().item())
        
    return centers
                            

@torch.no_grad()
def kmeans_pp(data, k, weights=None, device='cuda'):
    """
    Kmeans with ++ intialization:
    http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
     """
    
    n, d = data.size()
    centers = torch.empty(k, d, device=device)
    distances = torch.ones(n, device=device)
    for i in range(k):
        if weights is not None:
            distances *= weights
        distances = distances.clamp(min=1e-6)
        j = torch.multinomial(distances, 1).squeeze()
        centers[i] = data[j]
        distances, _ = get_distances(data, centers[:i + 1], device=device)      
            
    return centers
            
@torch.no_grad()
def kmeans_scalable(data, k, l, r=8, device='cuda'):
    """ 
    Scalable kmeans++: see 
    https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
    """
    
    n, d = data.size()
    centers = data[torch.multinomial(torch.ones(n, device=device), 1)
                    .squeeze()].view(1, d).to(device=device)
    
    for _ in range(r):
        distances, _ = get_distances(data, centers, device=device)
        p = l * distances / distances.sum()
        p = p.clamp(min=0, max=1)
        p = p.to(device=data.device)
        indices = torch.bernoulli(p).nonzero().view(-1, 1).expand(-1, d)
        centers = torch.cat((centers, 
                             data.gather(0, indices).to(device=device)), 0)
        
    _, indices = get_distances(data, centers)
    weights = indices.bincount(minlength=len(centers)).to(dtype=torch.float)
    
    centers = lloyd(centers, 
                    kmeans_pp(centers, k, weights, device=device), 
                    weights, 
                    device=device)
    
    return centers
        
@torch.no_grad()
def kmeans(data, k, device='cuda'):
    centers = kmeans_scalable(data, k, 4 * k, device=device)
    centers = lloyd(data, centers, device=device)
    return centers