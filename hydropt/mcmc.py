import pymc as pm
import arviz as az


def clear_nat_water(*args, wb):
    H2O_IOP = interpolate_to_wavebands(
        H2O_IOP_DEFAULT,
        wavelength=wb
    )
    return H2O_IOP.T.values


def phytoplankton(*args, wb):
    chl = args[0]
    # basis vector - according to Ciotti&Cullen (2002)

    a_phyto_base = interpolate_to_wavebands(
        data=a_phyto_base_full, 
        wavelength=wb
    )
    a = 0.06 * chl * a_phyto_base.absorption.values
    # constant spectral backscatter with backscatter ratio of 1.4%
    bb = np.repeat(.014 * 0.18 * chl, len(a))

    return np.array([a, bb])


def cdom(*args, wb):
    # absorption at 440 nm
    a_440 = args[0]

    # slope = 0.017
    # spectral absorption
    a = np.array(np.exp(-0.017 * (wb - 440)))

    # no backscatter
    bb = np.zeros(len(a))

    return a_440 * np.array([a, bb])


def nap(*args, wb):
    '''
    IOP model for NAP
    '''
    spm = args[0]
    # slope = args[1]


    # Absoprtion
    a = (.041 * .75 * np.exp(-.0123 * (wb - 443)))

    slope = 0.14 * 0.57
    bb = slope * (550 / wb)
    
    return spm * np.array([a, bb]) 


sample = run_df.iloc[575]

cols = [
    run_df.columns[i] 
    for i, col in enumerate(run_df.columns) 
    if 'Rrs' in col
]

wls = []
for col in cols:
    match = re.search('(\d*)$', col).group(0)
    wls.append(int(match))

wls = np.array(wls)
wl_use = ((wls >= 400) & (wls <= 700))
wls = wls[wl_use]

bio_opt = hd.BioOpticalModel()
bio_opt.set_iop(
    wavebands=wls,
    water=clear_nat_water,
    phyto=phytoplankton,
    cdom=cdom,
    nap=nap,
)
fwd_model = hd.PolynomialForward(bio_opt)

iops = fwd_model.iop_model.sum_iop([1, .4, 10])
rfl = fwd_model.refl_model.forward(iops)

test = hd.PolynomialReflectance()._polynomial_features(rfl)


with pm.Model() as model:
    phyto = pm.Uniform("phyto", lower=0.01, upper=200)
    cdom = pm.Uniform("cdom", lower=0.0005, upper=.2)
    nap = pm.Uniform("nap", lower=0.01, upper=300)

    iops = fwd_model.iop_model.sum_iop([phyto, cdom, nap])
    rfl = pm.Deterministic(
        "rfl",
        fwd_model.forward([phyto, cdom, nap]),
        dims="concentrations"
    )

with model:
    prior_samples = pm.sample_prior_predictive(100)


plt.plot(wls, iops[0, :])
plt.plot(wls, iops[1, :])
plt.show()


def invert_mcmc(self, y, x):

    row = sample.drop(['GLORIA_ID', 'N'])
    y = row.values[wl_use].astype(float)
    args = self._band_model((y, self._fwd_model.forward))

    with pm.Model() as model:
        phyto = pm.Uniform("phyto", lower=0.01, upper=200)
        cdom = pm.Uniform("cdom", lower=0.0005, upper=.2)
        nap = pm.Uniform("nap", lower=0.01, upper=300)

        loss_fun = lambda x, y, f: self._loss(dict(x.valuesdict()), y, f, w)
