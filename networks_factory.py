from typing import Sequence, Tuple
from brax.training import distribution
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from jax import numpy as jp
import jax
import numpy as np
from networks_base import make_policy_vae, make_value_network
import networks_base as networks

@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.VAE
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution
  
def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = policy_network.apply(*params, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample)
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions)
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions
      }

    return policy

  return make_policy

  
def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    encoder_layer_sizes: Sequence[int] = (1024, 1024),
    decoder_layer_sizes: Sequence[int] = (1024),
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    ) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  policy_network = make_policy_vae(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      encoder_layer_sizes=encoder_layer_sizes,
      decoder_layer_sizes=decoder_layer_sizes,
      )
  value_network = make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution)