"""Small PPO extension used by the Go2W residual MoE policy.

The regularizer is added to the PPO objective while the router is evaluated
for the same mini-batch. Keeping this in a local algorithm class avoids
patching the installed RSL-RL package and keeps old non-MoE tasks unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

# Huber transition point for the critic, in reward units.  The course rewards are
# sparse one-off events (+2.5 foothold, +-15 completion/termination), so a handful
# of returns sit far outside the bulk.  Squaring them lets a single sample own the
# critic gradient: in plum_step5 the value loss went 33 -> 2.4e3 -> 1.3e4 -> 1.6e5
# and the run was aborted.  Huber keeps the same minimiser for inliers and caps the
# per-sample contribution outside delta, which is the half of BeamDojo's
# double-critic benefit that needs no second network.
VALUE_HUBER_DELTA = 20.0


class MoERegularizedPPO(PPO):
    """PPO with a differentiable entropy/load-balancing router term."""

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_moe_loss = 0.0
        num_moe_updates = 0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = nn.functional.huber_loss(
                    values, batch.returns, delta=VALUE_HUBER_DELTA, reduction="none"
                )
                value_losses_clipped = nn.functional.huber_loss(
                    value_clipped, batch.returns, delta=VALUE_HUBER_DELTA, reduction="none"
                )
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = nn.functional.huber_loss(
                    values, batch.returns, delta=VALUE_HUBER_DELTA, reduction="mean"
                )

            moe_loss = torch.zeros((), device=self.device)
            get_moe_loss = getattr(self.actor, "get_moe_auxiliary_loss", None)
            if get_moe_loss is not None:
                moe_loss = get_moe_loss()
                num_moe_updates += 1

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + moe_loss
            )

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size])
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_moe_loss += moe_loss.detach().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if num_moe_updates:
            loss_dict["moe_auxiliary"] = mean_moe_loss / num_moe_updates
        self.storage.clear()
        return loss_dict
