import numpy as np
import tensorflow as tf

def interpolate_dna_one_hots(baseline,dna_one_hot,msteps):
    msteps_x = msteps[:, tf.newaxis, tf.newaxis ]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(dna_one_hot, axis=0)
    delta = input_x - baseline_x
    dna_one_hots = baseline_x +  msteps_x * delta
    return dna_one_hots


def compute_gradients(dna_one_hots,mymodel,label):
    with tf.GradientTape() as tape:
        tape.watch(dna_one_hots)
        logits = mymodel(dna_one_hots)
        #probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        if label ==1:
            probs=logits[0]
        elif label ==0:
            probs=1 - logits[0]
    return tape.gradient(probs, dna_one_hots)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


@tf.function
def one_batch(baseline, dna_one_hot, mstep_batch,mymodel,label):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_dna_one_hots(baseline=baseline,dna_one_hot=dna_one_hot,msteps=mstep_batch)
    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(dna_one_hots=interpolated_path_input_batch,mymodel=mymodel,label=label)
    return gradient_batch


def integrated_gradients(dna_one_hot,mymodel,label,baseline_freq=[0.3,0.3,0.2,0.2], m_steps=100,batch_size=64):
    assert len(dna_one_hot.shape)==2, "input must be 2 dims, e.g. (300,4)"
    assert dna_one_hot.shape[0]==300, "input 1st dim must be 300 (300bp)"
    dna_one_hot=tf.cast(dna_one_hot,tf.float32)
    baseline=tf.ones((300,4))
    baseline=baseline * baseline_freq
    
    # Generate msteps.
    msteps = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    # Collect gradients.        
    gradient_batches = []
    # Iterate msteps range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for mstep in tf.range(0, len(msteps), batch_size):
        from_ = mstep
        to = tf.minimum(from_ + batch_size, len(msteps))
        mstep_batch = msteps[from_:to]
        gradient_batch = one_batch(baseline, dna_one_hot, mstep_batch,mymodel,label)
        gradient_batches.append(gradient_batch)
    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)
    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(total_gradients)
    # Scale integrated gradients with respect to input.
    integrated_gradients = (dna_one_hot - baseline) * avg_gradients
    return integrated_gradients

"""
def integrated_gradients_other_layer(input_array,mymodel,baseline_freq,m_steps=100,batch_size=64):

    #compute grad from other layers
    #baseline_freq is required 
    #[0,0,].... size of input dim

    input_array=tf.cast(input_array,tf.float32)
    baseline=tf.ones(input_array.shape)
    baseline=baseline * baseline_freq
    
    # Generate msteps.
    msteps = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    # Collect gradients.        
    gradient_batches = []
    # Iterate msteps range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for mstep in tf.range(0, len(msteps), batch_size):
        from_ = mstep
        to = tf.minimum(from_ + batch_size, len(msteps))
        mstep_batch = msteps[from_:to]
        gradient_batch = one_batch(baseline, input_array, mstep_batch,mymodel)
        gradient_batches.append(gradient_batch)
    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)
    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(total_gradients)
    # Scale integrated gradients with respect to input.
    integrated_gradients = (input_array - baseline) * avg_gradients
    return integrated_gradients
"""
#import sys
#sys.path.append("/home/zhl022/daima/projects/dl_methods_improve/integrated_gradients/")
#import ig
#ig.integrated_gradients(fetal_one_hot[0,:300,:],model,label,m_steps=500)