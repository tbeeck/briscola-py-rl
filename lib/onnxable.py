import torch as th
from typing import Tuple

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class OnnxableMaskableACPolicy(th.nn.Module):
    def __init__(
        self,
        policy: MaskableActorCriticPolicy,
        share_features_extractor=True,
    ):
        super().__init__()
        self.policy = policy
        self.share_features_extractor = share_features_extractor

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.policy.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.policy.value_net(latent_vf)
        distribution = self.policy._get_action_dist_from_latent(latent_pi)
        action_likelihoods = distribution.distribution.probs

        return action_likelihoods, values
