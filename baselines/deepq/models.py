import tensorflow as tf


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_shape, num_actions):
        # the sub Functional model which does not include the top layer.
        model = network(input_shape)

        # wrapping the sub Functional model with layers that compute action scores into another Functional model.
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]

        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            action_out = latent
            for hidden in hiddens:
                action_out = tf.keras.layers.Dense(units=hidden, activation=None)(action_out)
                if layer_norm:
                    action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                action_out = tf.nn.relu(action_out)
            action_scores = tf.keras.layers.Dense(units=num_actions, activation=None)(action_out)

        if dueling:
            with tf.name_scope("state_value"):
                state_out = latent
                for hidden in hiddens:
                    state_out = tf.keras.layers.Dense(units=hidden, activation=None)(state_out)
                    if layer_norm:
                        state_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(state_out)
                    state_out = tf.nn.relu(state_out)
                state_score = tf.keras.layers.Dense(units=1, activation=None)(state_out)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return tf.keras.Model(inputs=model.inputs, outputs=[q_out])

    return q_func_builder

def build_pred_func(hiddens=[256], layer_norm=False, **network_kwargs):
    from baselines.common.models import get_network_builder
    convnet_i = get_network_builder('conv_only')(**network_kwargs)
    convnet_m = get_network_builder('conv_only')(**network_kwargs)
    mlp_a = get_network_builder('mlp_general')(**network_kwargs)
    deconvnet_iu = get_network_builder('deconv')(**network_kwargs)
    deconvnet_ic = get_network_builder('deconv')(**network_kwargs)
    deconvnet_m = get_network_builder('deconv')(**network_kwargs)
    convnet_a = get_network_builder('conv_mlp')(**network_kwargs)

    def image_func_builder(input_shape, latent_shape, num_actions):
        convnet_i_model = convnet_i(input_shape)
        obs_hist_latent = convnet_i_model.outputs[0]
        mlp_a_model = mlp_a(num_actions) 
        action_emb = mlp_a_model.outputs[0]

        deconvnet_iu_model = deconvnet_iu(latent_shape)
        iu = deconvnet_iu_model(obs_hist_latent)
        deconvnet_ic_model = deconvnet_ic((7, 7, 72))
        action_emb = tf.keras.layers.Reshape((7, 7, 8))(action_emb)
        concat_latent = tf.keras.layers.Concatenate()([obs_hist_latent, action_emb])
        ic = deconvnet_ic_model(concat_latent)

        return tf.keras.Model(inputs=convnet_i_model.inputs + mlp_a_model.inputs, 
            outputs=[iu, ic])
    
    def mask_func_builer(input_shape, latent_shape):
        convnet_m_model = convnet_m(input_shape)
        obs_curr_latent = convnet_m_model.outputs[0]

        deconvnet_m_model = deconvnet_m(latent_shape)
        m = deconvnet_m_model(obs_curr_latent)

        return tf.keras.Model(inputs=convnet_m_model.inputs, outputs=[m])

    def action_pred_func_builder(input_shape, num_actions):
        a_pred_model = convnet_a((84, 84, 2), num_actions)
        a_prob = a_pred_model.outputs[0]

        return tf.keras.Model(inputs=a_pred_model.inputs, outputs=a_prob)

    return [image_func_builder, mask_func_builer, action_pred_func_builder]