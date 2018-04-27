from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import data


# Set the random number generators' seeds for consistency
SEED = 9876
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)



def my_get_minibatches_idx(n, shuffle=False):
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    return zip(range(len(idx_list)), idx_list)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, _p, trng):
    '''proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.8, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 1.0)'''

    proj = state_before * trng.binomial(state_before.shape,
                                        p=_p, n=1,
                                        dtype=state_before.dtype)
    return proj




def dropout_mask_1D(state, dim, _p, trng):
    return trng.binomial(size=(state.shape[dim],), p=_p, n=1, dtype=state.dtype)


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options, Wemb_value=None):
    """
    Global (not NN) parameter. For the embeding and the classifier.
    """

    rng = numpy.random.RandomState(7896)

    params_d = OrderedDict()
    params_g = OrderedDict()

    # embeddings of cue types
    params_d['CueTemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_cueType'])),
                            high = numpy.sqrt(6. / (options['dim_cueType'])),
                            size=(options['n_cueTypes'], options['dim_cueType'])
                            )
                        )
                    ).astype(theano.config.floatX)

    # embeddings of differences of locations
    params_d['Lemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_locDiff'])),
                            high = numpy.sqrt(6. / (options['dim_locDiff'])),
                            size=(options['n_locDiffs'], options['dim_locDiff'])
                            )
                        )
                    ).astype(theano.config.floatX)




    # embeddings of tokens in sentences
    if Wemb_value is None:
        params_d['Wemb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['dim_token'])),
                            high = numpy.sqrt(6. / (options['dim_token'])),
                            size=(options['n_vocb_words'], options['dim_token'])
                            )
                        )
                    ).astype(theano.config.floatX)
    else:
        params_d['Wemb'] = (numpy.asarray(Wemb_value)).astype(theano.config.floatX)

    # embeddings to index
    params_g['to_idx_emb'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['lstm_decoder_layer_n'] + options['n_vocb_words'])),
                            high = numpy.sqrt(6. / (options['lstm_decoder_layer_n'] + options['n_vocb_words'])),
                            size=(options['lstm_decoder_layer_n'], options['n_vocb_words'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params_g['label_emb_0'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['ydim0'])),
                            high = numpy.sqrt(6. / (options['ydim0'])),
                            size=(options['ydim0'], options['dim_latent'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params_g['label_emb_1'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['ydim1'])),
                            high = numpy.sqrt(6. / (options['ydim1'])),
                            size=(options['ydim1'], options['dim_latent'])
                            )
                        )
                    ).astype(theano.config.floatX)




    #NN parameters
    params_d = param_init_lstm_0(options, params_d)
    params_g = param_init_lstm_decoder_layer(options, params_g)


    # classifier softmax
    params_d['Ws0'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['lstm_layer_n'] * 2 + options['lstm_layer_n'] * 0 + options['ydim0'])),
                            high = numpy.sqrt(6. / (options['lstm_layer_n'] * 2 + options['lstm_layer_n'] * 0 + options['ydim0'])),
                            size=(options['lstm_layer_n'] * 2 + options['lstm_layer_n'] * 0, options['ydim0'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params_d['bs0'] = numpy.zeros((options['ydim0'],)).astype(config.floatX)



    params_d['Ws1'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['n2'] + options['ydim1'])),
                            high = numpy.sqrt(6. / (options['n2'] + options['ydim1'])),
                            size=(options['n2'], options['ydim1'])
                            )
                        )
                    ).astype(theano.config.floatX)

    params_d['bs1'] = numpy.zeros((options['ydim1'],)).astype(config.floatX)


    params_d['Ws_fake'] = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (options['lstm_layer_n'] + 2)),
                            high = numpy.sqrt(6. / (options['lstm_layer_n'] + 2)),
                            size=(options['lstm_layer_n'], 2)
                            )
                        )
                    ).astype(theano.config.floatX)


    return [params_d, params_g]


'''def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params'''


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams



def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)



def init_my_param(size):

    rng = numpy.random.RandomState(7896)

    p = (numpy.asarray(rng.uniform(
                            low = -numpy.sqrt(6. / (size[0] + size[1])),
                            high = numpy.sqrt(6. / (size[0] + size[1])),
                            size = size))
                    ).astype(theano.config.floatX)

    return p


def param_init_lstm_0(options, myparams, prefix='lstm_layer'):
    """
    Init the LSTM_0 parameter:

    :see: init_params
    """

    rng = numpy.random.RandomState(6789)

    
    size_0 = (options['dim_token'], options['lstm_layer_n'])
    size_1 = (options['lstm_layer_n'],  options['lstm_layer_n'])

    myparams[_p(prefix, 'Wf')] = numpy.concatenate([init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0)], axis=1)

    myparams[_p(prefix, 'Uf')] = numpy.concatenate([init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1)], axis=1)

    myparams[_p(prefix, 'bf')] = numpy.zeros((options['lstm_layer_n'] * 4,)).astype(config.floatX)

    myparams[_p(prefix, 'Wb')] = numpy.concatenate([init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0),
                                                  init_my_param(size_0)], axis=1)

    myparams[_p(prefix, 'Ub')] = numpy.concatenate([init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1),
                                                  init_my_param(size_1)], axis=1)

    myparams[_p(prefix, 'bb')] = numpy.zeros((options['lstm_layer_n'] * 4,)).astype(config.floatX)



    myparams[_p(prefix, 'V')] = rng.normal(scale=0.01, size=(options['lstm_layer_n'] * 1,)).astype(config.floatX)
    ################################################################################################


    return myparams







def lstm_layer(tparams, input_state, mask, options, prefix='lstm_layer'):

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step_f(m_, x_, h_, c_):
        preact = tensor.dot(x_, tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')] + \
                 tensor.dot(h_, tparams[_p(prefix, 'Uf')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def _step_b(m_, x_, h_, c_):
        preact = tensor.dot(x_, tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')] + \
                 tensor.dot(h_, tparams[_p(prefix, 'Ub')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


    dim_proj = options[_p(prefix, 'n')]
    ##############################################################################################
    rval_f, updates_f = theano.scan(_step_f,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=input_state.shape[0])

    rval_b, updates_b = theano.scan(_step_b,
                                sequences=[mask, input_state],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[1], dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=input_state.shape[0],
                                go_backwards=True)


    proj_0 = rval_f[0] + rval_b[0][::-1]

    # Attention
    y_0 = (tensor.tanh(proj_0) * mask[:, :, None]) * tparams[_p(prefix, 'V')]
    y_0 = y_0.sum(axis=2).transpose()
    alpha = tensor.nnet.softmax(y_0).transpose()
    proj_0 = proj_0 * alpha[:, :, None]#(proj_0 * mask[:, :, None])

    proj_0 = proj_0.sum(axis=0)#(proj_0 * mask[:, :, None])
    ##############################################################################################


    proj_0 = tensor.tanh(proj_0)
    
     
    return proj_0





def param_init_lstm_decoder_layer(options, pg, prefix='lstm_decoder_layer'):


    rng = numpy.random.RandomState(6789)

    
    size_0 = (options['dim_latent'], options['lstm_decoder_layer_n'])
    size_1 = (options['lstm_decoder_layer_n'],    options['lstm_decoder_layer_n'])
    #size_2 = (options['lstm_decoder_layer_n'],    options['dim_latent'])

    pg[_p(prefix, 'W')] = numpy.concatenate([init_my_param(size_0),
                                                 init_my_param(size_0),
                                                 init_my_param(size_0),
                                                 init_my_param(size_0)], axis=1)

    pg[_p(prefix, 'U')] = numpy.concatenate([init_my_param(size_1),
                                                 init_my_param(size_1),
                                                 init_my_param(size_1),
                                                 init_my_param(size_1)], axis=1)


    pg[_p(prefix, 'b')] = numpy.zeros((options['lstm_decoder_layer_n'] * 4,)).astype(config.floatX)

    #pg[_p(prefix, 'W_x')] = init_my_param(size_2)

    #pg[_p(prefix, 'b_x')] = numpy.zeros((options['dim_latent'] * 1,)).astype(config.floatX)
    ################################################################################################

    return pg




def lstm_decoder_layer(tparams_all, input_state, options, maxlen, dp, prefix="lstm_decoder_layer"):

    tparams_d = tparams_all[0]
    tparams_g = tparams_all[1]

    #rng = numpy.random.RandomState(4567)
    trng = RandomStreams(SEED)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(x_, m_, h_, c_):

        preact = tensor.dot(x_, tparams_g[_p(prefix, 'W')]) + tparams_g[_p(prefix, 'b')] + \
                 tensor.dot(h_, tparams_g[_p(prefix, 'U')])
        
        i = tensor.nnet.sigmoid(_slice(preact, 0, options[_p(prefix, 'n')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options[_p(prefix, 'n')]))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options[_p(prefix, 'n')]))
        c = tensor.tanh(_slice(preact, 3, options[_p(prefix, 'n')]))

        c = f * c_ + i * c
        
        h = o * tensor.tanh(c)

        s = tensor.nnet.softmax(tensor.dot(h, tparams_g['to_idx_emb']))

        #x_t = tensor.dot((s / s.max(axis=1)[:,None]).astype('int32').astype(theano.config.floatX), tparams_d['Wemb'])
        x_t = tensor.dot(tensor.switch(s < s.max(axis=1)[:,None], 0.0, 1.0).astype(theano.config.floatX), 
                         tparams_d['Wemb'])

        x_out = s.argmax(axis=1)

        m = tensor.switch(tensor.eq(x_out, 10), 0.0, 1.0).astype(theano.config.floatX) * m_
        
        #x_t = tensor.dot(h_, tparams[_p(prefix, 'W_x')]) + tparams[_p(prefix, 'b_x')]

        return x_out, x_t, m, h, c


    ##############################################################################################
    rval, updates = theano.scan(_step,
                                outputs_info=[None,
                                              input_state,
                                              tensor.alloc(numpy_floatX(1.), input_state.shape[0]),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[0], options['lstm_decoder_layer_n']),
                                              tensor.alloc(numpy_floatX(0.), input_state.shape[0], options['lstm_decoder_layer_n'])],
                                name=_p(prefix, '_layers'),
                                n_steps=maxlen)


    #proj_0 = rval[1]#tensor.tanh(rval[0])

    m22 = trng.binomial(size=(input_state.shape[0],), p=dp, n=1, dtype=theano.config.floatX)
    
    #return rval[0]*m2, rval[1]*m2[:,None], rval[2]*m2

    if(tensor.gt(maxlen, 4) == 1):
        x2 = tensor.alloc(numpy.asarray(0, dtype='int32'), maxlen - 4, input_state.shape[0])
        x2 = tensor.concatenate((tensor.alloc(numpy.asarray(options['end_idx'], dtype='int32'), input_state.shape[0])[None, :],
                                 tensor.alloc(numpy.asarray(options['end_idx'], dtype='int32'), input_state.shape[0])[None, :],
                                 tensor.alloc(numpy.asarray(7, dtype='int32'), input_state.shape[0])[None, :],
                                 tensor.alloc(numpy.asarray(10, dtype='int32'), input_state.shape[0])[None, :],
                                 x2),
                                 axis=0)


        m2 = tensor.alloc(numpy_floatX(0.), maxlen - 3, input_state.shape[0])
        m2 = tensor.concatenate((tensor.alloc(numpy_floatX(1.), input_state.shape[0])[None, :],
                                 tensor.alloc(numpy_floatX(1.), input_state.shape[0])[None, :],
                                 tensor.alloc(numpy_floatX(1.), input_state.shape[0])[None, :],
                                 m2), 
                                 axis=0)
    
        xt2 = tparams_d['Wemb'][x2]

        return rval[0]*m22+x2*(1-m22), rval[1]*m22[:,None]+xt2*(1-m22[:,None]), rval[2]*m22+m2*(1-m22)

    else:
        return rval[0]*m22, rval[1]*m22[:,None], rval[2]*m22

    







def sgd(lr, tparams, grads, x, masks, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x + masks + y, 
                                    cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, masks, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(x + masks + y, 
                                    cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, masks, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(x + masks + y, 
                                    cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update




def momentum(lr, tparams, grads, x, masks, y, cost):
    
    rho = 0.9

    pre_step = [theano.shared(p.get_value() * 0., name='%s_pre_step' % k, 
                broadcastable = p.broadcastable)
               for k, p in tparams.items()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x + masks + y, 
                                    cost, 
                                    updates=gsup,
                                    name='momentum_sgd_f_grad_shared')


    step = [rho * ps - lr * g for ps, g in zip(pre_step, gshared)]

    sup = [(ps, s) for ps, s in zip(pre_step, step)]

    pup = [(p, p + s) for p, s in zip(tparams.values(), step)]


    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup+sup,
                               name='sgd_f_update')

    return f_grad_shared, f_update



def nesterov_momentum(lr, tparams, grads, x, masks, y, cost):
    
    rho = 0.9

    pre_step = [theano.shared(p.get_value() * 0., name='%s_pre_step' % k, 
                broadcastable = p.broadcastable)
               for k, p in tparams.items()]

    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x + masks + y, 
                                    cost, 
                                    updates=gsup,
                                    name='sgd_f_grad_shared')


    step = [rho * ps - lr * g for ps, g in zip(pre_step, gshared)]

    sup = [(ps, s) for ps, s in zip(pre_step, step)]

    pup = [(p, p + rho * rho * s - (1+rho) * lr * g) for p, s, g in zip(tparams.values(), step, gshared)]


    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup+sup,
                               name='sgd_f_update')

    return f_grad_shared, f_update






def Build_Model(tparams_all, options):

    trng = RandomStreams(SEED)

    # Discriminator
    x0 = tensor.matrix('x0', dtype='int32') #SIP Path
    x1 = tensor.matrix('x1', dtype='int32') #RelSrc Path
    x3 = tensor.matrix('x3', dtype='int32') #Cue Path

    mask0 = tensor.matrix('mask0', dtype=config.floatX)
    mask1 = tensor.matrix('mask1', dtype=config.floatX)
    mask3 = tensor.matrix('mask3', dtype=config.floatX)

    x0_d_y_fake = tensor.vector('x0_d_y_fake', dtype='int32')
    x1_d_y_fake = tensor.vector('x1_d_y_fake', dtype='int32')
    x3_d_y_fake = tensor.vector('x3_d_y_fake', dtype='int32')

    y0 = tensor.vector('y0', dtype='int32')
    y1 = tensor.vector('y1', dtype='int32')

    # Generator
    x_noise_0 = tensor.matrix('x_noise_0', dtype=config.floatX)
    x_noise_1 = tensor.matrix('x_noise_1', dtype=config.floatX)
    x_noise_3 = tensor.matrix('x_noise_3', dtype=config.floatX)

    #x0_g_y_fake = tensor.vector('x0_g_fake', dtype='int32')
    #x1_g_y_fake = tensor.vector('x1_g_fake', dtype='int32')
    #x3_g_y_fake = tensor.vector('x3_g_fake', dtype='int32')

    maxlen_0 = tensor.scalar(name='maxlen_0', dtype='int32')
    maxlen_1 = tensor.scalar(name='maxlen_1', dtype='int32')
    maxlen_3 = tensor.scalar(name='maxlen_3', dtype='int32')

    ###################

    dropout_ratio = tensor.scalar(name='dropout_ratio')
    dropout_decay_ratio = tensor.scalar(name='dropout_decay_ratio')

    tparams_d = tparams_all[0]
    tparams_g = tparams_all[1]

    #####################################
    # Discriminator
    p_0 = lstm_layer(tparams_d, input_state=tparams_d['Wemb'][x0], mask=mask0, options=options)
    p_1 = lstm_layer(tparams_d, input_state=tparams_d['Wemb'][x1], mask=mask1, options=options)
    p_3 = lstm_layer(tparams_d, input_state=tparams_d['Wemb'][x3[2:,:]], mask=mask3, options=options)

    proj_0 = tensor.concatenate((p_0, p_1), axis=1)
    proj_1 = tensor.concatenate((tparams_d['CueTemb'][x3[0, :]], 
                                 tparams_d['Lemb'][x3[1, :]], 
                                 p_3), axis=1)

    proj_0 = proj_0 * dropout_mask_1D(proj_0, 1, dropout_ratio, trng) * dropout_decay_ratio
    proj_1 = proj_1 * dropout_mask_1D(proj_1, 1, dropout_ratio, trng) * dropout_decay_ratio

    pred_0 = tensor.nnet.softmax(tensor.dot(proj_0, tparams_d['Ws0']) + tparams_d['bs0'])
    pred_1 = tensor.nnet.softmax(tensor.dot(proj_1, tparams_d['Ws1']) + tparams_d['bs1'])

    x0_d_fake_pred = tensor.nnet.softmax(tensor.dot(p_0, tparams_d['Ws_fake']))
    x1_d_fake_pred = tensor.nnet.softmax(tensor.dot(p_1, tparams_d['Ws_fake']))
    x3_d_fake_pred = tensor.nnet.softmax(tensor.dot(p_3, tparams_d['Ws_fake']))


    f_D_pred_prob = theano.function(inputs=[x0, x1, x3, mask0, mask1, mask3, dropout_ratio, dropout_decay_ratio], 
                                    outputs=[pred_0.max(axis=1), pred_1.max(axis=1)], 
                                    name='f_D_pred_prob')

    f_D_pred = theano.function(inputs=[x0, x1, x3, mask0, mask1, mask3, dropout_ratio, dropout_decay_ratio],
                               outputs=[pred_0.argmax(axis=1), pred_1.argmax(axis=1)], 
                               name='f_D_pred')

    off = 1e-8


    d_cost = - 1./3.*tensor.mean(tensor.log(x0_d_fake_pred[tensor.arange(x0_d_y_fake.shape[0]), x0_d_y_fake] + off)) + \
             - 1./3.*tensor.mean(tensor.log(x1_d_fake_pred[tensor.arange(x1_d_y_fake.shape[0]), x1_d_y_fake] + off)) + \
             - 1./3.*tensor.mean(tensor.log(x3_d_fake_pred[tensor.arange(x3_d_y_fake.shape[0]), x3_d_y_fake] + off)) + \
             - 1./2.*tensor.mean(tensor.log(pred_0[tensor.arange(y0.shape[0]), y0] + off)) + \
             - 1./2.*tensor.mean(tensor.log(pred_1[tensor.arange(y1.shape[0]), y1] + off))

    ##############################################################
    # Generator
    xn_0 = x_noise_0 * tparams_g['label_emb_0'][y0] * tparams_g['label_emb_1'][y1]
    xn_1 = x_noise_1 * tparams_g['label_emb_0'][y0] * tparams_g['label_emb_1'][y1]
    xn_3 = x_noise_3 * tparams_g['label_emb_0'][y0] * tparams_g['label_emb_1'][y1]

    x0_g, x0_g_emb, x0_g_mask = lstm_decoder_layer(tparams_all, xn_0, options, maxlen_0, 0.9)
    x1_g, x1_g_emb, x1_g_mask = lstm_decoder_layer(tparams_all, xn_1, options, maxlen_1, 0.9)
    x3_g, x3_g_emb, x3_g_mask = lstm_decoder_layer(tparams_all, xn_3, options, maxlen_3, 0.7)

    p_g0 = lstm_layer(tparams_d, input_state=x0_g_emb, mask=x0_g_mask, options=options)
    p_g1 = lstm_layer(tparams_d, input_state=x1_g_emb, mask=x1_g_mask, options=options)
    p_g3 = lstm_layer(tparams_d, input_state=x3_g_emb, mask=x3_g_mask, options=options)

    f_G_produce = theano.function(inputs=[x_noise_0, x_noise_1, x_noise_3, 
                                          maxlen_0, maxlen_1, maxlen_3,
                                          y0, y1],
                                  outputs=[x0_g.astype('int32'), x1_g.astype('int32'), x3_g.astype('int32'),
                                           x0_g_mask, x1_g_mask, x3_g_mask],
                                  name='f_G_produce')

    g_cost = (((p_0 - p_g0)**2).sum(axis=1).mean() + 
              ((p_1 - p_g1)**2).sum(axis=1).mean() + 
              ((p_3 - p_g3)**2).sum(axis=1).mean()) / 3.


    return [x0,x1,x3],[mask0, mask1, mask3],[x0_d_y_fake, x1_d_y_fake, x3_d_y_fake], [y0, y1], \
           [x_noise_0, x_noise_1, x_noise_3], \
           [maxlen_0, maxlen_1, maxlen_3], \
           f_D_pred_prob, f_D_pred, f_G_produce, \
           [dropout_ratio, dropout_decay_ratio], \
           d_cost, g_cost








def output_pred_labels(options, f_pred, f_pred_prob, prepare_data, data, iterator, verbose, path):
    f = open(path,'w')
    for _, test_index in iterator:
        x0, mask0, _ = prepare_data([data[0][t] for t in test_index])
        x1, mask1, _ = prepare_data([data[1][t] for t in test_index])
        x3, mask3, _ = prepare_data([data[2][t] for t in test_index], addIdxNum=2)
       
        
        pred_labels = f_pred(x0, x1,  x3, mask0, mask1,  mask3, 1., options['dropout_p'])

        pred_maxProbs = f_pred_prob(x0, x1,  x3, mask0,  mask1, mask3, 1., options['dropout_p'])


        for i in range(pred_labels[0].shape[0]):
            f.write(str(pred_labels[0][i])+' '+str(pred_labels[1][i])+' '+
                    str(pred_maxProbs[0][i])+' '+str(pred_maxProbs[1][i])+'\n')

    f.close()



def train_nn(

    # Hyper-Parameters

    dim_token = 100,  # word embeding dimension
    dim_locDiff = 10, # location difference dimension
    dim_cueType = 10, #
    dim_ESP_label = 10,
    dim_latent = 100,

    lstm_layer_n = 50,
    lstm_decoder_layer_n = 50,

    n2 = 50 + 10 + 10,
    ydim0 = 3,
    ydim1 = 3,
    #win_size = 2,

    #maxTokens1 = 60, # maximum tokens in sentence 1

    #n_ESP_labels = 3,
    n_cueTypes = 4,
    n_vocb_words = 4136,  # Vocabulary size
    n_locDiffs = 111,  # Location difference size

    end_idx = 3194,

    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=100,  # The maximum number of epoch to run
    #dispFreq=10,  # Display to stdout the training progress every N updates
    #decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    dropout_p = 1.0,
    
    optimizer = momentum,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).


    #maxlen=1000,  # Sequence longer then this get ignored
    batch_size=10,  # The batch size during training.
    inter_cost_margin = 0.001,


    # Parameter for extra option
    #noise_std=0.,
    #use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    #reload_model=None,  # Path to a saved model we want to start from.
    #test_size=-1
):

    

    # Model options
    model_options = locals().copy()
    print('-------------------------------------------------------------')
    print("model options", model_options)
    print('-------------------------------------------------------------')

    #load_data, prepare_data = get_dataset(dataset)

    print('Loading data ... ... ...')
    train, valid, test = data.load_data(path='../mydata.pkl.gz',n_words=n_vocb_words)
   
    

    print('Building model ... ... ...')

    params_all = init_params(model_options, Wemb_value=data.read_gz_file("../../matrix.pkl.gz"))

    
    #tparams = init_tparams(params)
    tparams_d = init_tparams(params_all[0])
    tparams_g = init_tparams(params_all[1])

   

    tparams_c = OrderedDict()
    for kk, pp in tparams_d.items():
        tparams_c[kk] = tparams_d[kk]

    for kk, pp in tparams_g.items():
        tparams_c[kk] = tparams_g[kk]


    
    (x,
     masks,
     x_d_y_fake,
     y,
     x_noises,
     x_maxlens,
     f_D_pred_prob,
     f_D_pred,
     f_G_produce,
     dropouts,
     d_cost,
     g_cost) = Build_Model([tparams_d, tparams_g], model_options)



   

    d_grads = tensor.grad(d_cost, wrt=list(tparams_d.values()))
    #print(tparams_c)
    g_grads = tensor.grad(g_cost, wrt=list(tparams_c.values()), consider_constant=list(tparams_d.values()),
        disconnected_inputs='ignore')
        

    lr = tensor.scalar(name='lr')

    #f_grad_shared, f_update = optimizer(lr, tparams, grads, x, masks, y, cost)

    f_D_grad_shared, f_D_update = optimizer(lr, tparams_d, d_grads,
                                            x + dropouts, masks, x_d_y_fake + y, d_cost)
    #f_G_grad_shared, f_G_update = optimizer(lr, tparams_c, g_grads,
                                            #x_noise + x_maxlen + x_d_ps + dropouts_g, [], x_g_y_fake + yg, g_cost)
    f_G_grad_shared, f_G_update = optimizer(lr, tparams_c, g_grads,
                                            x + x_noises + x_maxlens, masks, y, g_cost)


    print('training ... ... ...')

    kf_valid = get_minibatches_idx(len(valid[0]), batch_size)
    kf_test  = get_minibatches_idx(len(test[0]), batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    #history_errs = []
    best_p = None
    bad_counter = 0
    stop_counter = 0

    #if validFreq == -1:
        #validFreq = len(train[0]) // batch_size
    #if saveFreq == -1:
        #saveFreq = len(train[0]) // batch_size

    #last_training_sum_costs = numpy.inf
    last_ave_of_g_costs = numpy.inf
    last_ave_of_d_costs = numpy.inf

    g_costs_list = []
    d_costs_list = []


    uidx = 0  # the number of update done
    estop = False  # early stop
    #start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            #kf = get_minibatches_idx(99, batch_size, shuffle=True)


            #training_sum_costs = 0

            #ave_of_g_costs_sum = 0
            #ave_of_d_costs_sum = 0

            for train_batch_idx, train_index in kf:
                #uidx += 1
                #use_noise.set_value(1.)

                cur_batch_size = len(train_index)

                
                # Select the random examples for this minibatch
                x_0 = [train[0][t] for t in train_index]
                x_1 = [train[1][t] for t in train_index]

                x_3 = [train[2][t] for t in train_index]

                y_0 = [train[3][t] for t in train_index]
                y_1 = [train[4][t] for t in train_index]

                y_one_out = [train[5][t] for t in train_index]


                
                x_0, mask_0, maxlen_0 = data.prepare_data(x_0)
                x_1, mask_1, maxlen_1 = data.prepare_data(x_1)
                x_3, mask_3, maxlen_3 = data.prepare_data(x_3, addIdxNum=2)
                
                y_0 = numpy.asarray(y_0, dtype='int32')
                y_1 = numpy.asarray(y_1, dtype='int32')
                y_one_out = numpy.asarray(y_one_out, dtype='int32')

                rng = numpy.random.RandomState(9998)
                x0_noise_0 = rng.normal(scale=0.01, size=(cur_batch_size, dim_latent)).astype(config.floatX)
                x1_noise_1 = rng.normal(scale=0.01, size=(cur_batch_size, dim_latent)).astype(config.floatX)
                x3_noise_3 = rng.normal(scale=0.01, size=(cur_batch_size, dim_latent)).astype(config.floatX)


                generated_xs = f_G_produce(x0_noise_0, x1_noise_1, x3_noise_3,
                                          maxlen_0, maxlen_1, maxlen_3,
                                          y_0, y_1)


                #numpy.asarray([3] * cur_batch_size, dtype='int32')#
                generated_x_0 = generated_xs[0]
                generated_x_1 = generated_xs[1]
                generated_x_3 = numpy.concatenate(
                                    (#numpy.random.randint(0, n_cueTypes, (cur_batch_size,)).astype('int32')[None,:],
                                     #numpy.random.randint(0, n_locDiffs, (cur_batch_size,)).astype('int32')[None,:],
                                     x_3[0:2, :],
                                     generated_xs[2]), axis=0)
                generated_m_0 = generated_xs[3]
                generated_m_1 = generated_xs[4]
                generated_m_3 = generated_xs[5]
                generated_y_0 = numpy.random.randint(0, ydim0 - 1, (cur_batch_size,)).astype('int32')
                generated_y_1 = numpy.random.randint(0, ydim1, (cur_batch_size,)).astype('int32')

                
                x_d_0 = numpy.concatenate((x_0, generated_x_0), axis=1)
                x_d_1 = numpy.concatenate((x_1, generated_x_1), axis=1)
                
                x_d_3 = numpy.concatenate((x_3, generated_x_3), axis=1)
                y_d_0_fake = numpy.asarray([1] * cur_batch_size + [0] * cur_batch_size, dtype='int32')
                y_d_1_fake = numpy.asarray([1] * cur_batch_size + [0] * cur_batch_size, dtype='int32')
                y_d_3_fake = numpy.asarray([1] * cur_batch_size + [0] * cur_batch_size, dtype='int32')
                #mask_ones_0 = numpy.ones_like(mask_0)
                #mask_ones_1 = numpy.ones_like(mask_1)
                #mask_ones_3 = numpy.ones_like(mask_3)
                mask_d_0 = numpy.concatenate((mask_0, generated_m_0), axis=1)
                mask_d_1 = numpy.concatenate((mask_1, generated_m_1), axis=1)
                mask_d_3 = numpy.concatenate((mask_3, generated_m_3), axis=1)
                y_d_0 = numpy.concatenate((y_0, generated_y_0), axis=0)
                y_d_1 = numpy.concatenate((y_1, generated_y_1), axis=0)



               



                d_cost = f_D_grad_shared(x_d_0, x_d_1, x_d_3,
                                         dropout_p, 1.0,
                                         mask_d_0, mask_d_1, mask_d_3,
                                         y_d_0_fake, y_d_1_fake, y_d_3_fake, 
                                         y_d_0, y_d_1)

                g_cost = f_G_grad_shared(x_0, x_1, x_3,
                                         x0_noise_0, x1_noise_1, x3_noise_3,
                                         16, 16, 12,
                                         mask_0, mask_1, mask_3,
                                         generated_y_0, generated_y_1)

                #print(y_g_0.shape)

                print('\rd_cost = %f   g_cost = %f  @  %d' % (d_cost, g_cost, train_batch_idx), end='')
                #print(cur_batch_size)

                #ave_of_g_costs_sum += g_cost
                #ave_of_d_costs_sum += d_cost

                g_costs_list.append(g_cost)
                d_costs_list.append(d_cost)


                if d_cost < g_cost * 0.8:
                    for i in range(10):
                        f_G_update(0.01)
                        g_cost = f_G_grad_shared(x_0, x_1, x_3,
                                                 x0_noise_0, x1_noise_1, x3_noise_3,
                                                 16, 16, 12,
                                                 mask_0, mask_1, mask_3,
                                                 generated_y_0, generated_y_1)
                        if d_cost / g_cost >= 0.8 and  d_cost / g_cost <= 1.0/0.8:
                            break
                elif g_cost < d_cost * 0.8:
                    for i in range(10):
                        f_D_update(0.01)
                        d_cost = f_D_grad_shared(x_d_0, x_d_1, x_d_3,
                                         dropout_p, 1.0,
                                         mask_d_0, mask_d_1, mask_d_3,
                                         y_d_0_fake, y_d_1_fake, y_d_3_fake, 
                                         y_d_0, y_d_1)
                        if g_cost / d_cost >= 0.8 and g_cost / d_cost <= 1.0/0.8:
                            break
                else:
                    f_D_update(0.01)
                    f_G_update(0.01)



                if train_batch_idx % 100 == 0 or train_batch_idx == len(kf) - 1:
                    print("---Now %d/%d training bacthes @ epoch = %d" % (train_batch_idx, len(kf), eidx))

                if train_batch_idx > 0 and \
                   (train_batch_idx % 500 == 0 or train_batch_idx == len(kf) - 1):

                    cur_ave_of_d_costs = sum(d_costs_list) / len(d_costs_list)
                    cur_ave_of_g_costs = sum(g_costs_list) / len(g_costs_list)

                    print('ave_of_d_costs_sum = %f\tave_of_g_costs_sum = %f' % (cur_ave_of_d_costs, cur_ave_of_g_costs))            

                    #print('outputing predicted labels of test set ... ... ...')

                    output_pred_labels(model_options,
                               f_D_pred, f_D_pred_prob,
                               data.prepare_data, test, kf_test, 
                               verbose=False, path="test_pred_labels.txt")

                    if cur_ave_of_d_costs >=  last_ave_of_d_costs * 0.99 and \
                       cur_ave_of_g_costs >=  last_ave_of_g_costs * 0.99:
                        stop_counter += 1

                    last_ave_of_d_costs = cur_ave_of_d_costs
                    last_ave_of_g_costs = cur_ave_of_g_costs

                    print('counter for early stopping : %d/%d' % (stop_counter, patience))

                    del d_costs_list[:]
                    del g_costs_list[:]

                    if stop_counter >= patience:
                        print('Early Stop!')
                        estop = True
                        break

                

                #end for

                

           


            if stop_counter >= patience:
                print('Early Stop!')
                estop = True
                break

            if estop:
                break


    except KeyboardInterrupt:
        print("Training interupted")



if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_nn()
