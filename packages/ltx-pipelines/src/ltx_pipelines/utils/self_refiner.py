import torch


class PnPHandler:
    def __init__(self, stochastic_plan, ths_uncertainty=0.0, p_norm=1, certain_percentage=0.999, channel_dim: int = 1):
        self.stochastic_step_map = self._build_stochastic_step_map(stochastic_plan)
        self.ths_uncertainty = ths_uncertainty
        self.p_norm = p_norm
        self.certain_percentage = certain_percentage
        self.channel_dim = channel_dim
        self.buffer = [None]
        self.certain_flag = False

    def _build_stochastic_step_map(self, plan):
        step_map = {}
        if not plan:
            return step_map

        for entry in plan:
            if isinstance(entry, dict):
                start = entry.get("start", entry.get("begin"))
                end = entry.get("end", entry.get("stop"))
                steps = entry.get("steps", entry.get("anneal", entry.get("num_anneal_steps", 1)))
            else:
                start, end, steps = entry

            start_i = int(start)
            end_i = int(end)
            steps_i = int(steps)

            if steps_i > 0:
                for idx in range(start_i, end_i + 1):
                    step_map[idx] = steps_i
        return step_map

    def get_anneal_steps(self, step_index):
        return self.stochastic_step_map.get(step_index, 0)

    def reset_buffer(self):
        self.buffer = [None]
        self.certain_flag = False

    def process_step(self, latents, noise_pred, sigma, sigma_next, generator=None, device=None, latents_next=None, pred_original_sample=None):
        if pred_original_sample is None:
            pred_original_sample = latents - sigma * noise_pred

        if latents_next is None:
            latents_next = latents + (sigma_next - sigma) * noise_pred

        if self.buffer[-1] is not None:
            diff = pred_original_sample - self.buffer[-1][1]
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += latents.ndim
            uncertainty = torch.norm(diff, p=self.p_norm, dim=channel_dim) / latents.shape[channel_dim]

            certain_mask = uncertainty < self.ths_uncertainty
            if self.buffer[-1][0] is not None:
                certain_mask = certain_mask | self.buffer[-1][0]

            if certain_mask.sum() / certain_mask.numel() > self.certain_percentage:
                self.certain_flag = True

            certain_mask_float = certain_mask.to(latents.dtype).unsqueeze(channel_dim)

            latents_next = certain_mask_float * self.buffer[-1][2] + (1.0 - certain_mask_float) * latents_next
            pred_original_sample = certain_mask_float * self.buffer[-1][1] + (1.0 - certain_mask_float) * pred_original_sample

            certain_mask_stored = certain_mask
        else:
            certain_mask_stored = None
        self.buffer.append([certain_mask_stored, pred_original_sample, latents_next])
        return latents_next

    def perturb_latents(self, latents, buffer_latent, sigma, generator=None, device=None, noise_mask=None):
        noise = torch.randn(latents.shape, generator=generator, device=device or latents.device, dtype=latents.dtype)

        if noise_mask is None:
            return (1.0 - sigma) * buffer_latent + sigma * noise

        sigma_t = (noise_mask.to(latents.dtype) * sigma)
        return (1.0 - sigma_t) * buffer_latent + sigma_t * noise

    def run_refinement_loop(self,
                            latents,
                            noise_pred,
                            current_sigma,
                            next_sigma,
                            m_steps,
                            denoise_func,
                            step_func,
                            clone_func=None,
                            restore_func=None,
                            generator=None,
                            device=None,
                            noise_mask=None):

        if noise_pred is None:
            return None
        scheduler_state = None
        if clone_func:
            scheduler_state = clone_func()

        latents_next_0, pred_original_sample_0 = step_func(noise_pred, latents)
        if latents_next_0 is None or pred_original_sample_0 is None:
            return None

        latents_next = self.process_step(
            latents, noise_pred, current_sigma, next_sigma,
            latents_next=latents_next_0, pred_original_sample=pred_original_sample_0
        )

        if self.certain_flag:
            return latents_next

        for ii in range(1, m_steps):
            if restore_func and scheduler_state is not None:
                restore_func(scheduler_state)

            latents_perturbed = self.perturb_latents(
                latents,
                self.buffer[-1][1],
                current_sigma,
                generator=generator,
                device=device,
                noise_mask=noise_mask,
            )

            n_pred = denoise_func(latents_perturbed)
            if n_pred is None:
                return None

            latents_next_loop, pred_original_sample_loop = step_func(n_pred, latents_perturbed)
            if latents_next_loop is None or pred_original_sample_loop is None:
                return None

            latents_next = self.process_step(
                latents_perturbed, n_pred, current_sigma, next_sigma,
                latents_next=latents_next_loop, pred_original_sample=pred_original_sample_loop
            )

            if self.certain_flag:
                break

        return latents_next


def create_self_refiner_handler(pnp_plan, pnp_f_uncertainty, pnp_p_norm, pnp_certain_percentage, channel_dim: int = 1):
    stochastic_plan = None
    if isinstance(pnp_plan, list) and len(pnp_plan) > 0:
        stochastic_plan = pnp_plan

    if not stochastic_plan:
        stochastic_plan = [
            {"start": 2, "end": 8, "steps": 3},
        ]

    return PnPHandler(
        stochastic_plan,
        ths_uncertainty=pnp_f_uncertainty,
        p_norm=pnp_p_norm,
        certain_percentage=pnp_certain_percentage,
        channel_dim=channel_dim,
    )


def run_refinement_loop_multi(
    handlers,
    latents_list,
    noise_pred_list,
    current_sigma,
    next_sigma,
    m_steps,
    denoise_func,
    step_func,
    generators=None,
    devices=None,
    noise_masks=None,
    stop_when: str = "all",
):
    if m_steps <= 1:
        return latents_list
    if noise_pred_list is None:
        return None
    if not isinstance(noise_pred_list, (list, tuple)) or any(pred is None for pred in noise_pred_list):
        return None

    def _should_stop():
        if stop_when == "any":
            return any(handler.certain_flag for handler in handlers)
        return all(handler.certain_flag for handler in handlers)

    latents_next_list, pred_original_list = step_func(noise_pred_list, latents_list)
    if latents_next_list is None or pred_original_list is None:
        return None
    if not isinstance(latents_next_list, (list, tuple)) or not isinstance(pred_original_list, (list, tuple)):
        return None
    if len(latents_next_list) != len(handlers) or len(pred_original_list) != len(handlers):
        return None
    if any(latent is None for latent in latents_next_list) or any(pred is None for pred in pred_original_list):
        return None

    refined_latents_list = []
    for handler, latents, latents_next, pred_original in zip(
        handlers, latents_list, latents_next_list, pred_original_list
    ):
        refined_latents_list.append(
            handler.process_step(
                latents,
                None,
                current_sigma,
                next_sigma,
                latents_next=latents_next,
                pred_original_sample=pred_original,
            )
        )
    if _should_stop():
        return refined_latents_list

    for _ in range(1, m_steps):
        perturbed_list = []
        for idx, (handler, latents) in enumerate(zip(handlers, latents_list)):
            generator = generators[idx] if generators is not None else None
            device = devices[idx] if devices is not None else latents.device
            noise_mask = noise_masks[idx] if noise_masks is not None else None
            perturbed_list.append(
                handler.perturb_latents(
                    latents,
                    handler.buffer[-1][1],
                    current_sigma,
                    generator=generator,
                    device=device,
                    noise_mask=noise_mask,
                )
            )

        noise_pred_list = denoise_func(perturbed_list)
        if noise_pred_list is None:
            return None
        if not isinstance(noise_pred_list, (list, tuple)) or any(pred is None for pred in noise_pred_list):
            return None
        latents_next_list, pred_original_list = step_func(noise_pred_list, perturbed_list)
        if latents_next_list is None or pred_original_list is None:
            return None
        if not isinstance(latents_next_list, (list, tuple)) or not isinstance(pred_original_list, (list, tuple)):
            return None
        if len(latents_next_list) != len(handlers) or len(pred_original_list) != len(handlers):
            return None
        if any(latent is None for latent in latents_next_list) or any(pred is None for pred in pred_original_list):
            return None
        refined_latents_list = []
        for handler, latents, latents_next, pred_original in zip(
            handlers, perturbed_list, latents_next_list, pred_original_list
        ):
            refined_latents_list.append(
                handler.process_step(
                    latents,
                    None,
                    current_sigma,
                    next_sigma,
                    latents_next=latents_next,
                    pred_original_sample=pred_original,
                )
            )
        if _should_stop():
            break

    return refined_latents_list
