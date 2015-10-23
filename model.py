from collections import OrderedDict
from utils import norm_weight, get_layer, orth_weight

import copy

import numpy

import theano
from theano import tensor
from theano.tensor import concatenate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys

profile = False

# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])

    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])

    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    # decoder
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)

    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)

    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'], nout=options['dim_word'],
                                ortho=False)

    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)

    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'], nout=options['n_words'])
    return params

# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:,None]

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target)
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    proj_h = proj[0]
    ctxs = proj[1]
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                    prefix='ff_logit_ctx', activ='linear')

    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost

# build a sampler
def build_sampler(tparams, options, trng):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options, prefix='encoder')
    projr = get_layer(options['encoder'])[1](tparams, embr, options, prefix='encoder_r')
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
    ctx_mean = ctx.mean(0)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state)
    next_state = proj[0]
    ctxs = proj[1]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_lstm', activ='linear')

    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')

    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')

    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next

# generate sample
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0,nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[ti])
                new_hyp_states.append(copy.copy(next_state[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score

def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            import ipdb; ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)
