
import tensorflow as tf
print(('Using Tensorflow '+tf.__version__))
import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import time

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores

from collections import deque

# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

# read default parameters and override with custom ones
def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, filename, image, templates_z, scores, start_frame,
            cancel_threshold=None, freeze_template=False):
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames,4))

    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements    
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz

    # thresholds to saturate patches shrinking/growing
    min_z = hp.scale_min * z_sz
    max_z = hp.scale_max * z_sz
    min_x = hp.scale_min * x_sz
    max_x = hp.scale_max * x_sz

    # run_metadata = tf.RunMetadata()
    # run_opts = {
    #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #     'run_metadata': run_metadata,
    # }

    run_opts = {}

    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # save first frame position (from ground-truth)
        bboxes[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
        filename_ = frame_name_list[0]
        image_, new_templates_z_ = setup_template(sess, pos_x, pos_y, target_w, target_h, z_sz, image, filename_, templates_z, filename=filename)
        templates_z_ = new_templates_z_

        t_start = time.time()

        old_scores = deque(maxlen=30)

        # Get an image from the queue
        for i in range(1, num_frames):
            image_, bbox, new_score, new_scale_id, x_sz = track(sess, run_opts, hp, run, design, frame_name_list[i], pos_x, pos_y, target_w, target_h, x_sz, scale_factors, final_score_sz, penalty, filename, image, templates_z, templates_z_, scores)
            bboxes[i, :] = bbox

            # update the target representation with a rolling average
            if hp.z_lr > 0 and not freeze_template:
                templates_z_ = update_template(sess, image, image_, pos_x, pos_y, z_sz, templates_z, templates_z_, hp.z_lr)

            # update template patch size
            scaled_exemplar = z_sz * scale_factors
            z_sz = (1 - hp.scale_lr) * z_sz + hp.scale_lr * scaled_exemplar[new_scale_id]

            if cancel_threshold is not None:
                old_score = np.array(list(old_scores)).mean()
                if old_score is not None:
                    if new_score / old_score < cancel_threshold:
                        print("Cancelling old_score/new_score=%f < cancel_threshold=%f" % (
                            new_score / old_score, cancel_threshold))
                        break
                    print("Not cancelling old_score/new_score=%f > cancel_threshold=%f" % (
                        new_score / old_score, cancel_threshold))
                old_scores.append(new_score)


        t_elapsed = time.time() - t_start
        speed = num_frames/t_elapsed

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads) 

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline-search.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())

    plt.close('all')

    return bboxes, speed

def track(sess, run_opts, hp, run, design, image_, pos_x, pos_y, target_w, target_h, x_sz, scale_factors, final_score_sz, penalty, filename, image, templates_z, templates_z_, scores):
    scaled_search_area = x_sz * scale_factors
    scaled_target_w = target_w * scale_factors
    scaled_target_h = target_h * scale_factors

    image_, scores_ = sess.run(
        [image, scores],
        feed_dict={
            siam.pos_x_ph: pos_x,
            siam.pos_y_ph: pos_y,
            siam.x_sz0_ph: scaled_search_area[0],
            siam.x_sz1_ph: scaled_search_area[1],
            siam.x_sz2_ph: scaled_search_area[2],
            templates_z: np.squeeze(templates_z_),
            image: image_,
        }, **run_opts)
    scores_ = np.squeeze(scores_)
    # penalize change of scale
    scores_[0, :, :] = hp.scale_penalty * scores_[0, :, :]
    scores_[2, :, :] = hp.scale_penalty * scores_[2, :, :]
    # find scale with highest peak (after penalty)
    new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
    # update scaled sizes
    x_sz = (1 - hp.scale_lr) * x_sz + hp.scale_lr * scaled_search_area[new_scale_id]
    target_w = (1 - hp.scale_lr) * target_w + hp.scale_lr * scaled_target_w[new_scale_id]
    target_h = (1 - hp.scale_lr) * target_h + hp.scale_lr * scaled_target_h[new_scale_id]
    # select response with new_scale_id
    score_ = scores_[new_scale_id, :, :]

    my_score = score_

    # normalized scores
    score_ = score_ - np.min(score_)
    score_ = score_ / np.sum(score_)

    # apply displacement penalty
    score_ = (1 - hp.window_influence) * score_ + hp.window_influence * penalty
    pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz,
                                           hp.response_up, x_sz)

    bbox = pos_x, pos_y, target_w, target_h
    if run.visualization:
        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bbox_d = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
        show_frame(image_, bbox_d, 1)

    p = np.asarray(np.unravel_index(np.argmax(score_), np.shape(score_)))
    #print("Score bbox(%i,%i) %s max=%f(%f) at scale %d at %s" % (
    #    pos_x, pos_y, str(my_score.shape), np.max(my_score), np.max(score_), new_scale_id, str(p)))

    return image_, bbox, np.max(score_), new_scale_id, x_sz

def setup_template(sess, pos_x, pos_y, target_w, target_h, z_sz, image, image_, templates_z, filename=None):
    image_, templates_z_ = sess.run([image, templates_z], feed_dict={
        siam.pos_x_ph: pos_x,
        siam.pos_y_ph: pos_y,
        siam.z_sz_ph: z_sz,
        filename if filename is not None else image: image_})

    #print("templates_z_", templates_z_.shape)
    #import time
    #time.sleep(5)

    return image_, templates_z_

def update_template(sess, image, image_, pos_x, pos_y, z_sz, templates_z, templates_z_, lr):
    new_templates_z_ = sess.run([templates_z], feed_dict={
        siam.pos_x_ph: pos_x,
        siam.pos_y_ph: pos_y,
        siam.z_sz_ph: z_sz,
        image: image_
    })

    templates_z_ = (1 - lr) * np.asarray(templates_z_) + lr * np.asarray(new_templates_z_)
    return templates_z_


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


class Tracker:

    def __init__(self, hp, evaluation, run, env, design, graph, pos_x, pos_y, target_w, target_h, freeze_template=False):
        self.closed = False

        self.hp = hp
        self.evaluation = evaluation
        self.run = run
        self.env = env
        self.design = design

        self.freeze_template = freeze_template

        # Set size for use with tf.image.resize_images with align_corners=True.
        # For example,
        #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
        # instead of
        # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
        self.final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        # build TF graph once for all
        self.filename, self.image, self.templates_z, self.scores = graph
        self.image_ = None

        # save first frame position (from ground-truth)
        bbox = pos_x, pos_y, target_w, target_h
        self.track = [bbox]
        self.track_scores = [1e-8]

        self.scale_factors = hp.scale_step ** np.linspace(-np.ceil(hp.scale_num / 2), np.ceil(hp.scale_num / 2),
                                                     hp.scale_num)
        # cosine window to penalize large displacements
        hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        penalty = np.transpose(hann_1d) * hann_1d
        penalty = penalty / np.sum(penalty)
        self.penalty = penalty

        context = design.context * (target_w + target_h)
        self.z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        self.x_sz = float(design.search_sz) / design.exemplar_sz * self.z_sz

        self.templates_z_ = None

    def track_image(self, image_):
        pos_x, pos_y, target_w, target_h = self.track[-1]
        if self.image_ is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.global_variables_initializer().run(session=self.sess)
            # Coordinate the loading of image files.
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
            self.image_, self.templates_z_ = setup_template(self.sess, pos_x, pos_y, target_w, target_h, self.z_sz, self.image, image_,
                                                           self.templates_z)
            return self.track[-1], self.track_scores[-1]
        else:
            run_opts = {}
            image_, bbox, new_score, new_scale_id, self.x_sz = track(self.sess, run_opts, self.hp, self.run, self.design, image_, pos_x, pos_y,
                                                          target_w, target_h, self.x_sz, self.scale_factors, self.final_score_sz, self.penalty,
                                                          self.image, self.image, self.templates_z, self.templates_z_, self.scores)

            # update the target representation with a rolling average
            if self.hp.z_lr > 0 and not self.freeze_template:
                self.templates_z_ = update_template(self.sess, self.image, self.image_, pos_x, pos_y, self.z_sz, self.templates_z, self.templates_z_, self.hp.z_lr)

            # update template patch size
            scaled_exemplar = self.z_sz * self.scale_factors
            self.z_sz = (1 - self.hp.scale_lr) * self.z_sz + self.hp.scale_lr * scaled_exemplar[new_scale_id]

            self.track.append(bbox)
            self.track_scores.append(new_score)
            return bbox, new_score

    def close(self):
        # Finish off the filename queue coordinator.
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()
        self.closed = True

    def update_template(self, image_, bbox):
        if self.closed:
            return

        pos_x, pos_y, target_w, target_h = bbox
        context = self.design.context * (target_w + target_h)
        z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))

        self.templates_z_ = update_template(self.sess, self.image, image_, pos_x, pos_y, z_sz, self.templates_z, self.templates_z_, self.hp.z_lr)
