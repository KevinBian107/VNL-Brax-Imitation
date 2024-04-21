import dataclasses
from typing import Any, Callable, Sequence, Tuple
from brax.training import types
from brax.training.spectral_norm import SNDense
import jax
import jax.numpy as jnp
from jax import random
import warnings
from brax.training import types
from brax.training import distribution
from brax.training.networks import MLP
from flax import linen as nn

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]

class Decoder(nn.Module):
    '''DEcoder for VAE'''
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
        return x

class Encoder(nn.Module):
    '''Encoder for VAE'''
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    bias: bool = True
    latents: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # For each layer in the sequence
        # Make a dense net and apply layernorm then tanh
        for i, hidden_size in enumerate(self.layer_sizes):
            x = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(x)
            x = nn.LayerNorm(x)
            x = self.activation(x)
            
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        return mean_x, logvar_x

class VAE(nn.Module):
  """Full VAE model."""

  encoder_layers: Sequence[int]
  decoder_layers: Sequence[int]
  latents: int = 60

  def setup(self):
    self.encoder = Encoder(layer_sizes=self.encoder_layers, latents=self.latents)
    self.decoder = Decoder(layer_sizes=self.decoder_layers)

  # x = (traj_dims * traj_length + state_dims)
  def __call__(self, x, e_rng, z_rng):
    traj = x[:traj_dims * traj_length]
    state = x[traj_dims * traj_length:]
    mean, logvar = self.encoder(traj, e_rng)
    z = reparameterize(z_rng, mean, logvar)
    action = self.decoder(z.cat(state))
    return action, mean, logvar

  def generate(self, z):
    return self.decoder(z) + noise
  
def make_policy_vae(
    param_size: int,
    latent_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    encoder_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_layer_sizes: Sequence[int] = (1024),
    ) -> VAE:
  """Creates a policy VAE network."""
  
  policy_module = VAE(
      encoder_layers=list(encoder_layer_sizes) + [latent_size],
      decoder_layers=list(decoder_layer_sizes) + [param_size], 
      latents = param_size)

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = nn.relu) -> FeedForwardNetwork:
  """Creates a value network, like normal ones"""

  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)

class MLP_CNN(linen.Module):
  '''Simple MLP CNN Network for reference'''
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    #print(data.shape) # initial should all be zero

    # two dimension matrix slicing
    vision_data = data[...,data.shape[-1]-12288:] #just vision, ant have 27, rodent have more, change automatically with getting image of 12288
    pro_data = data[...,:data.shape[-1]-12288] #just proprioception

    dtype = jnp.float32
    vision_data = vision_data.astype(dtype) / 255.0
    #print(vision_data.shape)

    # vmap_size = -1 # automatically infered size #vision_data.shape[0]
    # vision_data = vision_data.reshape((vmap_size, 240, 320, 3)) # reshape back to 3d image with vmap considered

    #handling dynamic new shape issues
    #avoid error in case of 1 d as well, add anything that is not the [-1] position, extract all dimensions except the last one
    new_shape_prefix = vision_data.shape[:-1]
    new_shape = new_shape_prefix + (64, 64, 3)
    vision_data = vision_data.reshape(new_shape)
    print(f'Before into ConvNet + vmap shape: {vision_data.shape}')

    vision_data = linen.Conv(features=32,
                      kernel_size=(8, 8),
                      strides=(4, 4),
                      name='conv1',
                      dtype=dtype,
                      )(vision_data)
    vision_data = linen.relu(vision_data)
    vision_data = linen.Conv(features=64,
                      kernel_size=(4, 4),
                      strides=(2, 2),
                      name='conv2',
                      dtype=dtype,
                      )(vision_data)
    vision_data = linen.relu(vision_data)
    vision_data = linen.Conv(features=64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      name='conv3',
                      dtype=dtype,
                      )(vision_data)
    vision_data = linen.relu(vision_data)
    
    # flatten preserving expected dimension of 76800, then fit automaticall
    # vision_data = vision_data.reshape((-1, 76800))

    # fully connected layer
    vision_data = linen.Dense(features=512, name='hidden', dtype=dtype)(vision_data)
    vision_data = linen.relu(vision_data)
    vision_out = linen.Dense(features=1, name='logits', dtype=dtype)(vision_data) # this is (2560, 1)
    print(f'This is out of CovNet {vision_out.shape}')

    # handling dynamic new shape issues
    out_new_shape_prefix = pro_data.shape[:-1]
    out_new_shape = out_new_shape_prefix + (-1,)
    vision_out = vision_out.reshape(out_new_shape)

    hidden = jnp.concatenate([pro_data, vision_out], axis=-1)

    for i, hidden_size in enumerate(self.layer_sizes):
      # print(f'hidden_unit_size:{hidden_size}')
      # print(f'hidden_input_size:{hidden.shape}')    
      # hidden size is a integer [hidden_layer_size, parameter size (which is a NormalTanhDistribution)]

      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
          )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
      
    print(f'this is out of full ppo network {hidden.shape}')

    return hidden