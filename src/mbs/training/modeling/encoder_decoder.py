from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict, Literal, List
import pickle

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from .decoders import create_decoder, create_projector


def load_linear_probe(decoder: nn.Sequential, probe_path: Union[str, Path], freeze: bool = True) -> None:
    """Load saved sklearn ridge weights into a linear PyTorch decoder."""
    probe_path = Path(probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Linear probe file does not exist: {probe_path}")

    with probe_path.open("rb") as f:
        probe_model = pickle.load(f)

    linear = decoder.fc1
    weight = torch.as_tensor(probe_model["W"], dtype=linear.weight.dtype, device=linear.weight.device)
    bias = torch.as_tensor(probe_model["b"], dtype=linear.bias.dtype, device=linear.bias.device)

    if tuple(weight.shape) != tuple(linear.weight.shape):
        raise ValueError(
            f"Probe weight shape {tuple(weight.shape)} does not match decoder "
            f"shape {tuple(linear.weight.shape)} for {probe_path}"
        )
    if tuple(bias.shape) != tuple(linear.bias.shape):
        raise ValueError(
            f"Probe bias shape {tuple(bias.shape)} does not match decoder "
            f"shape {tuple(linear.bias.shape)} for {probe_path}"
        )

    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    for param in decoder.parameters():
        param.requires_grad = not freeze


class BrainEncoderDecoder(nn.Module):
    """
    A neural network model that combines an encoder and multiple decoders to process and predict neural responses.

    This model is designed to take both image and neural stimuli as inputs, process them through a shared encoder, and then use separate decoders to predict neural responses for different brain regions.

    Attributes:
        default_decoder (nn.Module): The default decoder to use if no specific decoder is provided for a feature.

    Args:
        encoder (Union[torch.fx.GraphModule, torch.nn.Module]): The encoder model that processes the inputs.
        decoders (Dict[str, nn.Module], optional): A dictionary mapping decoder labels to decoder models. Defaults to an empty dictionary.
        decoder_input_feats_map (Dict[str, str], optional): A dictionary mapping feature names to decoder labels. Defaults to an empty dictionary.

    Methods:
        forward(batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int = 0, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
            Forward pass of the model. Chooses between concatenated or separate input processing based on `concat_inputs`.

        _forward_separate(batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            Forward pass when separate inputs are used. Processes image and neural stimuli inputs separately through the encoder and decodes the features.
    """
    
    default_decoder = nn.Sequential(OrderedDict([
        ('decoder_input', nn.Identity()),
    ]))
    
    def __init__(self, 
                 encoder: Union[GraphModule, torch.nn.Module],
                 decoders: Dict[str, nn.Module] = {},
                 projectors: Dict[str, nn.Module] = {},
                 decoder_input_feats_map: Dict[str, str] = {},
                 proj_input_feats_map: Dict[str, str] = {},
                 decoder_type: Literal['whole_brain', 'region_specific']='whole_brain'
                 ):
        super(BrainEncoderDecoder, self).__init__()
        
        
        self.encoder = encoder
        self.decoder_input_feats_map = decoder_input_feats_map
        self.proj_input_feats_map = proj_input_feats_map
        self.decoder_type = decoder_type

        
        # Setup decoders. If no decoder is provided for a feature, use the default decoder.
        # Convert the decoders to a ModuleDict for easy access.
        self.decoders = {}
        for decoder_name in decoders.keys():
            self.decoders[decoder_name] = decoders.get(decoder_name, self.default_decoder)
        self.decoders = nn.ModuleDict(self.decoders)
        
        self.projectors = {}
        for proj_id, proj in projectors.items():
            self.projectors[proj_id] = proj
        self.projectors = nn.ModuleDict(self.projectors)

    def forward(
        self, 
        batch:Union[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]], 
        batch_idx:int=0, 
        dataloader_idx:int=0, 
        mode:Literal['tensor_input', 'image_only', 'neural_stimulus_only', 'separate']='tensor_input', 
        return_image_output_only:bool=True
        ) -> Dict[str, torch.Tensor]:
        if isinstance(batch, tuple):
            batch = batch[0]
        if mode == 'tensor_input':
            return self._forward_tensor(batch, return_image_output_only=return_image_output_only)
        elif mode == 'image_only':
            return self._forward_image(batch)
        elif mode == 'neural_stimulus_only':
            return self._forward_neural_stimulus(batch)
        elif mode == 'separate':
            return self._forward_separate(batch)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def _forward_tensor(self, input, return_image_output_only:bool=True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Standard forward pass for a model that takes a single tensor input. It expects a single tensor input and returns a single tensor output
        if return_image_output_only is True, otherwise it returns a dictionary of outputs. This is useful for contexts where the model is used as a feature extractor
        such as Brain-Score benchmarks.
        """
        
        features = self.encoder(input)
        
        outputs = {
            "output": features["output"]
        }

        if self.decoder_type == 'whole_brain':
            latent_features = [
                proj(latent_feature) 
                for proj, latent_feature in 
                zip(self.projectors.values(), features.values())
            ]
            projected_features = torch.cat(latent_features, dim=-1)
            features['whole_brain_feats'] = projected_features
            outputs['whole_brain_feats'] = features['whole_brain_feats']
        elif self.decoder_type == 'region_specific':
            for proj_name, proj in self.projectors.items():
                features[proj_name] = proj(features[self.proj_input_feats_map[proj_name]])
                outputs[proj_name] = features[proj_name]

        for decoder_label, input_feats_label in self.decoder_input_feats_map.items():
            outputs[decoder_label] = self.decoders[decoder_label](features[input_feats_label])
            
        if return_image_output_only:
            return outputs['output']
        return outputs
        

    def _forward_image(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = batch["input"]
        image_output = self.encoder(image)["output"]
        return {"output": image_output}
    
    def _forward_neural_stimulus(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        neural_stimuli = batch["neural_stimulus"]
        
        features = {}
        for subject in neural_stimuli.keys():
            features_subject = self.encoder(neural_stimuli[subject])
            for key, value in features_subject.items():
                features[f"{subject}_{key}"] = value
        # features = self.encoder(neural_stimuli)
        outputs = {}
        
        if self.decoder_type == 'whole_brain':
            raise NotImplementedError("This is legacy code path and is not implemented.")
            # latent_features = [
            #     proj(latent_feature) 
            #     for proj, latent_feature in 
            #     zip(self.projectors.values(), features.values())
            # ]
            # projected_features = torch.cat(latent_features, dim=-1)
            # features['whole_brain_feats'] = projected_features
            # outputs['whole_brain_feats'] = features['whole_brain_feats']
        elif self.decoder_type == 'region_specific':
            for proj_name, proj in self.projectors.items():
                features[proj_name] = proj(features[self.proj_input_feats_map[proj_name]])
                # outputs[proj_name] = features[proj_name]

        
        for decoder_label, input_feats_label in self.decoder_input_feats_map.items():
            outputs[decoder_label] = self.decoders[decoder_label](features[input_feats_label])
        return outputs
    
    def _forward_separate(self, batch:Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        image = batch["image"]["input"]
        neural_stimuli = batch["neural"]["neural_stimulus"]
        
        image_output = self.encoder(image)["output"]
        # features = self.encoder(neural_stimuli)
        features = {}
        for subject in neural_stimuli.keys():
            features_subject = self.encoder(neural_stimuli[subject])
            for key, value in features_subject.items():
                features[f"{subject}_{key}"] = value
        
        outputs = {}
        outputs["output"] = image_output

        if self.decoder_type == 'whole_brain':
            latent_features = [
                proj(latent_feature) 
                for proj, latent_feature in 
                zip(self.projectors.values(), features.values())
            ]
            projected_features = torch.cat(latent_features, dim=-1)
            features['whole_brain_feats'] = projected_features
            outputs['whole_brain_feats'] = features['whole_brain_feats']
        elif self.decoder_type == 'region_specific':
            for proj_name, proj in self.projectors.items():
                features[proj_name] = proj(features[self.proj_input_feats_map[proj_name]])
                # outputs[proj_name] = features[proj_name]

        for decoder_label, input_feats_label in self.decoder_input_feats_map.items():
            outputs[decoder_label] = self.decoders[decoder_label](features[input_feats_label])

        return outputs


def create_encoder_decoder(backbone: nn.Module, feat_layers:Dict[str,str]=None, **training_config) -> nn.Module:

    decoder_type = training_config.get('decoder_type', 'whole_brain')
    if decoder_type == 'whole_brain':
        # return_nodes = {f"backbone.{feat_layer}":f"feats_{feat_id}" for feat_id, feat_layer in enumerate(feat_layers.values())}
        return_nodes = {f"backbone.{feat_layer}":f"{feat_layer.replace('.','_')}_feats" for region_name, feat_layer in feat_layers.items()}
    elif decoder_type == 'region_specific':
        return_nodes = {f"backbone.{feat_layer}":f"{feat_layer.replace('.','_')}_feats" for region_name, feat_layer in feat_layers.items()}
        
    

    
    neural_encoding_layers = list(return_nodes.values())
    print(f"Using feature layers: {return_nodes}")

    # Get the names of the nodes in the computational graph
    train_nodes, eval_nodes = get_graph_node_names(backbone)
    penultimate_layer = train_nodes[-2]

    # Assign the last layer to the "neural_behavior" feature
    # return_nodes[penultimate_layer] = "neural_behavior_feats"

    # Map the region names to the corresponding layer names
    return_nodes["backbone"] = "output"
    

    # Create the feature extractor
    if 'cornet' in training_config['arch']:
        # Loops inside CORnet blocks are not supported by the default feature extractor
        from scaling_primate_vvs.training.src.models.cornet import CORblock_S
        tracer_kwargs = {'leaf_modules': [CORblock_S]}
    else:
        tracer_kwargs = None
    encoder = create_feature_extractor(backbone, return_nodes=return_nodes, tracer_kwargs=tracer_kwargs)


    # Get the dimensions of inputs to decoders
    n_channels = 3
    h, w = training_config.get('train_crop_size', 224), training_config.get('train_crop_size', 224)
    model_input_dims = (n_channels, h, w)
    inp = torch.randn(1, *model_input_dims)
    out = encoder(inp)
    # latent_feats_dims = {k: v.numel() for k,v in out.items()}
    latent_feats_dims = {k: v.shape for k,v in out.items()}
    
    subjects = training_config['data_neural_subjects']
    rois = training_config['rois']
    
    projectors = {}
    # projector_config = training_config.get('projectors', {})
    # proj_output_dim = projector_config.get('output_dim', 0)
    # if training_config.get('overwrite_proj_dim'):
    #     proj_output_dim = training_config['overwrite_proj_dim']
    # token_reduce_method = projector_config.get('token_reduction_method', None)
    # freeze_projection = projector_config.get('freeze_projection', True)
    # for feat_name in neural_encoding_layers:
        
    #         input_dim = latent_feats_dims[feat_name]
    #         projectors[feat_name] = create_projector(
    #             input_dim=input_dim, 
    #             output_dim=proj_output_dim, 
    #             token_reduce_method=token_reduce_method,
    #             freeze_projection=freeze_projection
    #         )
    
    proj_input_feats_map = {}
    for subj in subjects:
        for roi in rois:
            # proj_name = f"{subj}_{roi}"
            proj_name = f"{subj}_{roi}_feats"
            features_label = f"{feat_layers[roi].replace('.','_')}_feats"
            input_dim = latent_feats_dims[features_label]
            latent_feats_dims[proj_name] = input_dim
            projectors[proj_name] = nn.Identity()
            proj_input_feats_map[proj_name] = f"{subj}_{features_label}"
    

    # Define the decoders
    decoder_type = "region_specific"
    decoder_output_dims = training_config["decoder_output_dims"]
    linear_probes_dir = training_config.get('linear_probes_dir', None)
    decoder_input_feats_map = {}
    decoders = {}
    for subject in subjects:
        for roi in rois:

            # Get the decoder configuration
            decoder_label = f"subj_{subject}-roi_{roi}"
            features_label = f"{subject}_{roi}_feats"
            decoder_input_feats_map[decoder_label] = features_label
            num_hidden_layers = training_config.get('decoder_hidden_layers', 0)
            decoder_hidden_dim = training_config.get('decoder_hidden_dim', 512)
            
            input_dim = latent_feats_dims[features_label]
            input_dim = input_dim.numel()
                
            output_dim = decoder_output_dims[decoder_label]
            is_frozen = training_config.get('frozen_decoders', True)
            # Create the decoder
            decoders[decoder_label] = create_decoder(
                input_dim=input_dim, 
                output_dim=output_dim, 
                hidden_dim=decoder_hidden_dim, 
                num_hidden_layers=num_hidden_layers,
                is_frozen=is_frozen
            )
            if linear_probes_dir is not None and num_hidden_layers == 0:
                probe_path = Path(linear_probes_dir) / f"subject_{subject}" / f"roi_{roi}.pkl"
                load_linear_probe(decoders[decoder_label], probe_path, freeze=is_frozen)
                print(f"Loaded linear probe for subject {subject}, roi {roi} from {probe_path}")
            else:
                print(f"Created new decoder for subject {subject}, roi {roi}")
                
    for subject in subjects:
        for roi in rois:

            # Get the decoder configuration
            decoder_label = f"pearsonr_subj_{subject}-roi_{roi}"
            features_label = f"{subject}_{roi}_feats"
            decoder_input_feats_map[decoder_label] = features_label
            num_hidden_layers = 0
            decoder_hidden_dim = training_config.get('decoder_hidden_dim', 512)
            
            

            input_dim = latent_feats_dims[features_label]
            input_dim = input_dim.numel()

            output_dim = decoder_output_dims[decoder_label.replace('pearsonr_','')]
            # is_frozen = True
            is_frozen = training_config.get('frozen_decoders', True)
            # Create the decoder
            decoders[decoder_label] = create_decoder(
                input_dim=input_dim, 
                output_dim=output_dim, 
                hidden_dim=decoder_hidden_dim, 
                num_hidden_layers=num_hidden_layers,
                is_frozen=is_frozen
            )
            if linear_probes_dir is not None and num_hidden_layers == 0:
                probe_path = Path(linear_probes_dir) / f"subject_{subject}" / f"roi_{roi}.pkl"
                load_linear_probe(decoders[decoder_label], probe_path, freeze=is_frozen)
                print(f"Loaded linear probe for subject {subject}, roi {roi} from {probe_path}")
            else:
                print(f"Created new decoder for subject {subject}, roi {roi}")
                
                
    print(proj_input_feats_map)
    print(projectors)
    print(decoder_input_feats_map)
    print(decoders)

        

    brain_model = BrainEncoderDecoder(
        encoder=encoder, 
        decoders=decoders,
        projectors=projectors,
        decoder_input_feats_map=decoder_input_feats_map,
        proj_input_feats_map=proj_input_feats_map,
        decoder_type=decoder_type
    )
    # print(brain_model)

    return brain_model
