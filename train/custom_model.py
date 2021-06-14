from train.utils import GradientAccumulatorModel

import tensorflow as tf


class CustomDetectorModel(GradientAccumulatorModel):

    def __init__(self, detection_model, input_shape, num_grad_accum=1, **kargs):
        super(CustomDetectorModel, self).__init__(num_accum=num_grad_accum, **kargs)
        self.detection_model = detection_model
        self.this_input_shape = input_shape
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.cls_loss_tracker = tf.keras.metrics.Mean(name='cls_loss')
        self.loc_loss_tracker = tf.keras.metrics.Mean(name='loc_loss')

    def compile(self, **kargs):
        super(CustomDetectorModel, self).compile(**kargs)

    def call(self, inputs, training=False):
        image_tensors, shapes = inputs
        prediction_dict = self.detection_model.predict(image_tensors, shapes)
        return prediction_dict

    def train_step(self, data):
        imgs, labels, boxes = data
        labels = tf.one_hot(tf.cast(labels, tf.int32), 1)
        labels = [tf.expand_dims(x, 0) for x in labels]
        boxes = [tf.expand_dims(x, 0) for x in boxes]
        batch_size = len(labels)
        shapes = tf.constant(batch_size * [self.this_input_shape], dtype=tf.int32)
        self.detection_model.provide_groundtruth(
            groundtruth_boxes_list=boxes,
            groundtruth_classes_list=labels)
        with tf.GradientTape() as tape:
            prediction_dict = self([imgs, shapes], training=True)
            losses_dict = self.detection_model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.accumulate_grads_and_apply(grads)
        self.loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(losses_dict['Loss/classification_loss'])
        self.loc_loss_tracker.update_state(losses_dict['Loss/localization_loss'])
        return {'loss': self.loss_tracker.result(),
            self.cls_loss_tracker.name: self.cls_loss_tracker.result(),
            self.loc_loss_tracker.name: self.loc_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_loss_tracker, self.loc_loss_tracker]
