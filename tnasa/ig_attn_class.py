import numpy as np
import tensorflow as tf
class IgAttn:
    def __init__(self,
        input_array,
        mymodel,
        baseline_freq, 
        myvalue_full, 
        my_pre_input_full, 
        current_i,
        label,
        m_steps=100,
        batch_size=64):
        self.input_array=input_array
        self.mymodel=mymodel
        self.baseline_freq=baseline_freq
        self.myvalue_full=myvalue_full
        self.my_pre_input_full=my_pre_input_full
        self.current_i=current_i
        self.label=label
        self.m_steps=m_steps
        self.batch_size=batch_size

    def interpolate_dna_one_hots(self,baseline,dna_one_hot,msteps):
        msteps_x = msteps[:, tf.newaxis, tf.newaxis,tf.newaxis ]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(dna_one_hot, axis=0)
        delta = input_x - baseline_x
        dna_one_hots = baseline_x +  msteps_x * delta
        return dna_one_hots
    def compute_gradients(self,dna_one_hots,mymodel,label):
        #tf.config.run_functions_eagerly(True)
        with tf.GradientTape() as tape:
            tape.watch(dna_one_hots)
            logits = mymodel(dna_one_hots)
            #probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
            if label ==1:
                probs=2*tf.math.maximum(tf.subtract(logits,0.5) ,0.0)
            elif label ==0:
                probs=2*tf.math.maximum(tf.subtract(0.5,logits),0.0)
        return tape.gradient(probs, dna_one_hots)
    def integral_approximation(self,gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients


    @tf.function
    def one_batch(self,baseline, dna_one_hot, mstep_batch,mymodel,label):
        # Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = self.interpolate_dna_one_hots(baseline=baseline,dna_one_hot=dna_one_hot,msteps=mstep_batch)
        # Compute gradients between model outputs and interpolated inputs.
        gradient_batch = self.compute_gradients(dna_one_hots=interpolated_path_input_batch,mymodel=mymodel,label=label)
        return gradient_batch
    def integrated_gradients_attn_layer(self):
        """
        compute grad from other layers
        baseline_freq is required 
        [0,0,].... size of input dim
        """

        self.input_array=tf.cast(self.input_array,tf.float32)
        self.baseline=tf.ones(self.input_array.shape)
        self.baseline=self.baseline * self.baseline_freq
        self.mymodel.layers[1].myv = self.myvalue_full[self.current_i,...]
        self.mymodel.layers[2].pre_input=self.my_pre_input_full[self.current_i,...]

        # Generate msteps.
        msteps = tf.linspace(start=0.0, stop=1.0, num=self.m_steps+1)
        # Collect gradients.        
        gradient_batches = []
        # Iterate msteps range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for mstep in tf.range(0, len(msteps), self.batch_size):
            from_ = mstep
            to = tf.minimum(from_ + self.batch_size, len(msteps))
            mstep_batch = msteps[from_:to]
            gradient_batch = self.one_batch(self.baseline, self.input_array, mstep_batch,self.mymodel,label=self.label)
            gradient_batches.append(gradient_batch)
        # Concatenate path gradients together row-wise into single tensor.
        total_gradients = tf.concat(gradient_batches, axis=0)
        # Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(total_gradients)
        # Scale integrated gradients with respect to input.
        integrated_gradients = (self.input_array - self.baseline) * avg_gradients
        return integrated_gradients
    