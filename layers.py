import theano
import theano.tensor as tensor
import numpy

from utils import import prfx, norm_weight, ortho_weight

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[prfx(prefix,'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[prfx(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[prfx(prefix,'W')]) + tparams[prfx(prefix,'b')])

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[prfx(prefix,'W')] = W
        params[prfx(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[prfx(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[prfx(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[prfx(prefix,'Ux')] = Ux
    params[prfx(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[prfx(prefix,'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[prfx(prefix, 'W')]) + tparams[prfx(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[prfx(prefix, 'Wx')]) + tparams[prfx(prefix, 'bx')]
    U = tparams[prfx(prefix, 'U')]
    Ux = tparams[prfx(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                non_sequences = [tparams[prfx(prefix, 'U')],
                                                 tparams[prfx(prefix, 'Ux')]],
                                name=prfx(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval

# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[prfx(prefix,'Wc')] = Wc

    Wcx = norm_weight(dimctx,dim)
    params[prfx(prefix,'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin,dimctx)
    params[prfx(prefix,'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prfx(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[prfx(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prfx(prefix,'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx,1)
    params[prfx(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prfx(prefix, 'c_tt')] = c_att

    return params

def gru_cond_layer(tparams,
                   state_below,
                   options,
                   prefix='gru',
                   mask=None,
                   context=None,
                   one_step=False,
                   init_memory=None,
                   init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prfx(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[prfx(prefix,'Wc_att')]) + tparams[prfx(prefix,'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[prfx(prefix, 'Wx')]) + tparams[prfx(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[prfx(prefix, 'W')]) + tparams[prfx(prefix, 'b')]
    state_belowc = tensor.dot(state_below, tparams[prfx(prefix, 'Wi_att')])

    def _step_slice(m_, x_, xx_, xc_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wcx):
        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None,:,:]
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:,:,None]).sum(0) # current context

        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[prfx(prefix, 'U')],
                   tparams[prfx(prefix, 'Wc')],
                   tparams[prfx(prefix,'Wd_att')],
                   tparams[prfx(prefix,'U_att')],
                   tparams[prfx(prefix, 'c_tt')],
                   tparams[prfx(prefix, 'Ux')],
                   tparams[prfx(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info = [init_state,
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_,
                                                   context]+shared_vars,
                                    name=prfx(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
            state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
            state_before * 0.5)
    return proj
