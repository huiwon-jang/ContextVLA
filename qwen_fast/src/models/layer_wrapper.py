import torch.nn as nn
import torch

class LayerWrapper(nn.Module):
    def __init__(
            self, 
            layer, 
            layer_idx, 
            internal_projection=2, 
            num_frames=1, 
            num_views=1,
            index_context=None,
            img_pattern=[27, 1805, 220],
            motion_token=0
    ):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.internal_projection = internal_projection
        self.input_id_context = None
        self.num_frames = num_frames
        self.num_views = num_views
        self.index_context = index_context
        self.motion_token = motion_token
        self.img_pattern = img_pattern
        self.compressed_length = None

    def get_removing_indices(self, hidden_states):
        pat_len = len(self.img_pattern)

        windows = self.input_id_context.unfold(dimension=1, size=pat_len, step=1)
        pattern_tensor = torch.tensor(self.img_pattern, device=hidden_states.device).view(1, 1, -1)
        matches = (windows == pattern_tensor).all(dim=-1)
        match_indices = torch.nonzero(matches, as_tuple=True)[1].reshape(matches.shape[0], -1)[:, :self.num_views * (self.num_frames - 1) + 1]
        begin_idx = match_indices[:,0].view(-1, 1)
        end_idx   = match_indices[:,-1].view(-1, 1)

        return begin_idx, end_idx

    def forward(self, hidden_states, *args, **kwargs):
        if hidden_states.shape[1] > 1:
            self.position_ids = kwargs['position_ids']
            self.cache_position = kwargs['cache_position']
            self.position_embeddings = kwargs['position_embeddings']

            if self.layer_idx == self.internal_projection:
                bsz, seq_len, dim = hidden_states.shape
                device = hidden_states.device

                token_indices = torch.arange(seq_len, device=device).view(1, -1).expand(bsz, -1)
                begin_idx, end_idx = self.get_removing_indices(hidden_states) #need to remove [begin_idx, end_idx)

                keep_mask = (token_indices < begin_idx) | (token_indices >= end_idx)
                new_len = keep_mask.sum(dim=1)[0].item()

                batch_indices = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, new_len)
                gather_indices = token_indices[keep_mask].view(bsz, new_len)

                if self.motion_token > 0:
                    drop_mask = ~keep_mask
                    dropped_indices = token_indices[drop_mask].view(bsz, -1)
                    batch_indices_drop = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, dropped_indices.shape[-1])
                    hidden_dropped = hidden_states[batch_indices_drop, dropped_indices].reshape(bsz, self.motion_token, -1, hidden_states.shape[-1])
                    hidden_dropped = hidden_dropped.mean(dim=2)
                    hidden_states = torch.cat([hidden_dropped, hidden_states[batch_indices, gather_indices]], axis=1)
                else:
                    hidden_states = hidden_states[batch_indices, gather_indices]

                if 'attention_mask' in kwargs.keys() and kwargs['attention_mask'] is not None:
                    if self.motion_token > 0:
                        kwargs['attention_mask'] = torch.cat([
                            torch.ones(bsz, self.motion_token, device=kwargs['attention_mask'].device, dtype=kwargs['attention_mask'].dtype), 
                            kwargs['attention_mask'][batch_indices, gather_indices]], 
                        dim=1)
                    else:
                        kwargs['attention_mask'] = kwargs['attention_mask'][batch_indices, gather_indices]
                    
                if 'position_ids' in kwargs.keys() and kwargs['position_ids'] is not None:
                    kwargs['position_ids'] = kwargs['position_ids'][:,:,:new_len+self.motion_token]

                if 'cache_position' in kwargs.keys() and kwargs['cache_position'] is not None:
                    kwargs['cache_position'] = kwargs['cache_position'][:new_len+self.motion_token]
                if 'position_embeddings' in kwargs.keys() and kwargs['position_embeddings'] is not None:
                    kwargs['position_embeddings'] = (kwargs['position_embeddings'][0][:, :, :new_len+self.motion_token], kwargs['position_embeddings'][1][:, :, :new_len+self.motion_token])
                self.index_context.batch_indices = batch_indices
                self.index_context.gather_indices = gather_indices
                self.compressed_length = new_len + self.motion_token
            elif self.layer_idx > self.internal_projection:
                new_len = self.index_context.batch_indices.shape[1]
                if 'attention_mask' in kwargs.keys() and kwargs['attention_mask'] is not None:
                    if self.motion_token > 0:
                        kwargs['attention_mask'] = torch.cat([
                            torch.ones(hidden_states.shape[0], self.motion_token, device=kwargs['attention_mask'].device, dtype=kwargs['attention_mask'].dtype), 
                            kwargs['attention_mask'][self.index_context.batch_indices, self.index_context.gather_indices]], 
                        dim=1)
                    else:
                        kwargs['attention_mask'] = kwargs['attention_mask'][self.index_context.batch_indices, self.index_context.gather_indices]
                    
                if 'position_ids' in kwargs.keys() and kwargs['position_ids'] is not None:
                    kwargs['position_ids'] = kwargs['position_ids'][:,:,:new_len+self.motion_token]
                if 'cache_position' in kwargs.keys() and kwargs['cache_position'] is not None:
                    kwargs['cache_position'] = kwargs['cache_position'][:new_len+self.motion_token]
                if 'position_embeddings' in kwargs.keys() and kwargs['position_embeddings'] is not None:
                    kwargs['position_embeddings'] = (kwargs['position_embeddings'][0][:, :, :new_len+self.motion_token], kwargs['position_embeddings'][1][:, :, :new_len+self.motion_token])
                self.compressed_length = new_len + self.motion_token
        else:
            if self.layer_idx >= self.internal_projection:
                kwargs['position_ids'] = self.position_ids[:, :, self.compressed_length].unsqueeze(-1)
                kwargs['cache_position'] = self.cache_position[self.compressed_length]
                kwargs['position_embeddings'] = (self.position_embeddings[0][:, :, self.compressed_length].unsqueeze(-2), self.position_embeddings[1][:, :, self.compressed_length].unsqueeze(-2))
                self.compressed_length += 1

        return self.layer(hidden_states, *args, **kwargs)