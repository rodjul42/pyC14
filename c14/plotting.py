import numpy as np
import pandas as pd
import scipy as sp

from .tools import col_name_levels, listofdict_to_dictoflist, RK45,ImplicitParametersOutOfRange

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import logging
logger = logging.getLogger(__name__)
__all__ = ['visualisze']


class visualisze():
    def __init__(self, model, data=None, Dbirth_sim=np.arange(1900, 2019),
                 age=100, step_size=0.1, rtol=0.1, min_step=0.001):
        self.model = model
        self.Dbirth = Dbirth_sim
        self.age = age
        self.t_eval = np.arange(0, age, step_size)
        self.span = (0, age)
        self.N = len(self.Dbirth)
        self.C_init = self.model.gen_init(self.Dbirth)
        self.step_size = step_size
        self.rtol = 0.1
        self.rtol = rtol
        self.min_step = min_step
        self.data = data

        if self.data is not None:
            self.Dbirth_data = self.data.Dbirth
            self.t_eval_data = np.arange(0, age, step_size)
            self.span_data = (0, age)
            self.N_data = len(self.data.Dbirth)
            self.C_init_data = self.model.gen_init(self.data.Dbirth)

    def odeint(self):
        self.model.set_Dbirth(self.Dbirth)
        self.M_new = np.zeros(self.C_init.shape)
        if self.step_size is None:
            sol = sp.integrate.solve_ivp(
                    fun=self.model.rhs,
                    y0=self.C_init.unstack().values.copy(),
                    rtol=self.rtol,
                    t_span=self.span,
                    t_eval=self.t_eval,
                    min_step=self.min_step,
                    method=RK45)
        else:
            sol = sp.integrate.solve_ivp(
                    fun=self.model.rhs,
                    y0=self.C_init.unstack().values.copy(),
                    max_step=self.step_size,
                    atol=np.inf,
                    rtol=np.inf,
                    t_span=self.span,
                    t_eval=self.t_eval,
                    method='RK45')

        mcol = pd.MultiIndex.from_product([self.model.var_names, np.arange(self.N)],
                                          names=['variable', 'sample'])
        sol_res =  self.model.bins_to_value(sol['y'],self.N,len(self.t_eval),self.t_eval+self.Dbirth[:,np.newaxis])
        self.solpd = pd.DataFrame(np.reshape(sol_res, (self.model.nvars*self.N, -1)).T,
                                  index=sol['t'], columns=mcol)

        if self.data is not None:
            self.model.set_Dbirth(self.Dbirth_data)
            self.M_new = np.zeros(self.C_init_data.shape)
            if self.step_size is None:
                sol = sp.integrate.solve_ivp(
                        fun=self.model.rhs,
                        y0=self.C_init_data.unstack().values.copy(),
                        rtol=self.rtol,
                        t_span=self.span_data,
                        t_eval=self.t_eval_data,
                        min_step=self.min_step,
                        method=RK45)
            else:
                sol = sp.integrate.solve_ivp(
                        fun=self.model.rhs,
                        y0=self.C_init_data.unstack().values.copy(),
                        max_step=self.step_size, atol=np.inf, rtol=np.inf,
                        t_span=self.span_data,
                        t_eval=self.t_eval_data,
                        method='RK45')
            mcol = pd.MultiIndex.from_product([self.model.var_names, self.data.df.index], names=['variable', 'sample'])
            sol_res_data = self.model.bins_to_value(sol['y'],self.N_data,len(self.t_eval_data),self.t_eval_data+self.Dbirth_data[:,np.newaxis])
            self.solpd_data = pd.DataFrame(np.reshape(sol_res_data, (self.model.nvars*self.N_data, -1)).T, index=sol['t'], columns=mcol)
            return self.solpd, self.solpd_data
        else:
            return self.solpd

    def plot_parameter(self, parameters=None, errors=None, alpha=0.1,
                       axis=None, log=False, nonlog=None, no_plot=[]):
        if axis is None:
            axis = plt.gca()

        if log:
            axis.set_yscale('log')
        if nonlog is not None:
            twin_axis = axis.twinx()
        else:
            nonlog = []

        if parameters is None:
            plot_error = False
        else:
            self.model.set_parameters_fit(parameters)
            if errors is not None:
                plot_error = True
            else:
                plot_error = False

        t_start = self.t_eval[0]
        t_end = self.t_eval[-1]
        paras_colors = dict()
        for p_name in self.model.parameter_names:
            if p_name in no_plot:
                continue
            p = self.model.__dict__[p_name]
            if p_name in nonlog:
                lines = axis.plot([t_end+1, t_end+2], [p, p],
                                  label=p_name, ls='--', zorder=5)
                lcolor = lines[0].get_color()
                twin_axis.plot([t_start, t_end], [p, p],
                               color=lcolor, ls='--', zorder=5)
            else:
                lines = axis.plot([t_start, t_end], [p, p],
                                  label=p_name, zorder=5)
                lcolor = lines[0].get_color()
            paras_colors[p_name] = lcolor

        ip_time_idx, iparas = self.try_iparas()
        iparas_t = self.t_eval[ip_time_idx]
        iparas_names = list(iparas.keys())
        iparas_colors = dict()
        for p_name in iparas_names:
            if p_name in no_plot:
                continue
            p = iparas[p_name]
            if p_name in nonlog:
                lines = axis.plot([t_end+1, t_end+2], [p[0], p[-1]],
                                  label=p_name, ls='--', zorder=5)
                lcolor = lines[0].get_color()
                twin_axis.plot(iparas_t, p, color=lcolor, ls='--', zorder=5)
            else:
                lines = axis.plot(iparas_t, p, label=p_name, zorder=5)
                lcolor = lines[0].get_color()
            iparas_colors[p_name] = lcolor

        if plot_error:
            paras_errors, iparas_errors, ip_time = errors
            for p_name in self.model.parameter_names:
                if p_name in no_plot:
                    continue
                p_error = paras_errors[p_name]
                if p_name in nonlog:
                    twin_axis.fill_between([t_start, t_end],
                                           [p_error[1], p_error[1]],
                                           [p_error[0], p_error[0]],
                                           zorder=1,
                                           color=paras_colors[p_name],
                                           alpha=alpha)
                else:
                    axis.fill_between([t_start, t_end],
                                      [p_error[1], p_error[1]],
                                      [p_error[0], p_error[0]],
                                      zorder=1,
                                      color=paras_colors[p_name],
                                      alpha=alpha)

            for p_name in iparas_errors.keys():
                if p_name in no_plot:
                    continue
                ip_error = iparas_errors[p_name]

                if p_name in nonlog:
                    twin_axis.fill_between(ip_time, ip_error[1], ip_error[0],
                                           zorder=1,
                                           color=iparas_colors[p_name],
                                           alpha=alpha)
                else:
                    axis.fill_between(ip_time, ip_error[1], ip_error[0],
                                      zorder=1, color=iparas_colors[p_name],
                                      alpha=alpha)

        axis.set_xlim(t_start, t_end)
        axis.set_xlabel('age')
        plt.sca(axis)

    def try_iparas(self, t_eval=None,ignore_physics=False):
        if t_eval is None:
            t_eval = self.t_eval
        t_eval = np.atleast_1d(t_eval)
        ip_time_idx = []
        iparas_s = []
        for t_ind, t in enumerate(t_eval):
            try:
                tmp = self.model.get_implicit_parameters(t)
            except ImplicitParametersOutOfRange as e:
                if ignore_physics:
                    tmp = e.iparas
                    logger.info("Ignore unphysical value exception %s", e)
                else:
                    logger.info("Invaild implicit parameters for t=%s with %s", t, e)
                    continue
            except ValueError as e:
                logger.info('Model does not return iparas in exception. Fallback to ignore_physics=True,\n Invaild implicit parameters for t=%s with %s', t, e)
                continue
            ip_time_idx.append(t_ind)
            iparas_s.append(tmp)

        if len(iparas_s) == 0:
            raise ValueError('No iparas possible')

        iparas = {n: [] for n in iparas_s[0].keys()}
        for ip in iparas_s:
            for n in ip.keys():
                iparas[n].append(ip[n])
        return ip_time_idx, iparas

    def calc_error(self, parameters, cov, samples=1000, t_eval=None,
                   ignore_physics=False, confidence=0.682, seed=42):
        np.random.seed(seed)
        
        if 'sigma' not in parameters.keys():
            parameters['sigma'] = 0
            cov['sigma'] = 0
            cov.loc['sigma'] = 0
            cov.loc['sigma', 'sigma'] = 0

        if t_eval is None:
            t_eval = self.t_eval
        t_eval = np.atleast_1d(t_eval)
        percentile = 100*(1 - confidence)/2

        boot_p_fit = np.random.multivariate_normal(
                        [parameters[name]
                         for name in self.model.parameter_names],
                        (cov[self.model.parameter_names]
                         .loc[self.model.parameter_names].values),
                        size=samples)

        self.model.set_parameters_fit(parameters)

        # Filter boot_paras outside limits
        boot_p_fit = boot_p_fit[
            [self.model.set_parameters_fit_array(p_fit, ignore_physics)
             for p_fit in boot_p_fit]
            ]

        list_iparas = []
        valid_time_idx = set()
        for p_fit in boot_p_fit:
            self.model.set_parameters_fit_array(p_fit, ignore_physics)
            try:
                ip_time_idx, iparas = self.try_iparas(t_eval, ignore_physics)
                valid_time_idx = valid_time_idx.union(set(ip_time_idx))
            except ValueError as e:
                logger.info('Value exception: %s', e)
                continue
            list_iparas.append(iparas)

        boot_iparas = listofdict_to_dictoflist(list_iparas)
        valid_time_idx = list(valid_time_idx)

        # return boot_iparas, boot_paras

        iparas_error = dict()
        for name in boot_iparas.keys():
            ip_lower_error = []
            ip_upper_error = []
            for t_idx in valid_time_idx:
                tmp = []
                for i in boot_iparas[name]:
                    try:
                        tmp.append(i[t_idx])
                    except:
                        pass
                tmp = np.array(tmp)
                logger.debug('Bootstrapsamples for time %s parameter %s is %s',
                             t_idx, name, len(tmp))
                ip_lower_error.append(np.percentile(tmp, percentile))
                ip_upper_error.append(np.percentile(tmp, 100-percentile))
            iparas_error[name] = (ip_lower_error, ip_upper_error)

        paras_error = dict()
        boot_p_phy = np.array([[self.model.transform_fit_to_physical_array(p_fit)[name]
                                for name in self.model.parameter_names]
                               for p_fit in boot_p_fit])
        lower_error = {name: np.percentile(boot_p_phy[:, i], percentile)
                       for i, name in enumerate(self.model.parameter_names)}
        upper_error = {name: np.percentile(boot_p_phy[:, i], 100-percentile)
                       for i, name in enumerate(self.model.parameter_names)}
        paras_error = {name: (lower_error[name], upper_error[name])
                       for name in self.model.parameter_names}
        return paras_error, iparas_error, t_eval[valid_time_idx]

    def get_parameter(self, parameters, cov, samples=1000, t_eval=None,
                      ignore_physics=False, confidence=0.682, seed=42):
        if t_eval is None:
            t_eval = self.t_eval
        t_eval = np.atleast_1d(t_eval)

        errors = self.calc_error(parameters, cov, samples, t_eval,
                                 ignore_physics, confidence, seed=seed)
        t_eval = errors[2]
        self.model.set_parameters_fit(parameters)
        ip_time_idx, iparas = self.try_iparas(t_eval, ignore_physics)

        if not np.all(t_eval == t_eval[ip_time_idx]):
            raise Exception('could not calculate implicit parameters')

        parameter_names = list(errors[0].keys()) + list(errors[1].keys())
        result = pd.DataFrame(
            index=pd.MultiIndex.from_product([parameter_names, t_eval],
                                             names=['parameter', 'time']),
            columns=['point_estimate', 'lower', 'upper'])

        for p_name in self.model.parameter_names:
            result.loc[p_name, 'point_estimate'] = self.model.__dict__[p_name]

        for p_name in errors[0].keys():
            result.loc[p_name, 'lower'] = errors[0][p_name][0]
            result.loc[p_name, 'upper'] = errors[0][p_name][1]
        for p_name in errors[1].keys():
            result.loc[p_name, 'lower'] = np.array(errors[1][p_name][0])
            result.loc[p_name, 'upper'] = np.array(errors[1][p_name][1])
            result.loc[p_name, 'point_estimate'] = np.array(iparas[p_name])

        return result.swaplevel().sort_index()


        dfp = pd.DataFrame(result, index=t_eval)
        dfi = pd.DataFrame(iparas, index=t_eval)
        df = dfp.join(dfi)
        df.index.name = 'time'
        df.columns.name = 'parameter'
        return df.reindex(sorted(df.columns), axis=1)

    def plot_generic(self, variables=None, plot_D14C=True, figure=None,
                     cmap=None, sm=None):
        try:
            self.solpd
        except AttributeError:
            logging.info('run odeint first')
            return None, None
        if variables is None:
            variables = self.model.var_names
            nvars = len(variables)
        else:
            nvars = len(variables)
        plots = nvars
        if figure is None:
            figure = plt.figure(figsize=(8, 3*plots))
        shape = (plots, 1)
        axies = {var: plt.subplot2grid(shape, (var_i, 0))
                 for var_i, var in enumerate(variables)}

        if sm is None:
            if cmap is None:
                cmap = plt.cm.jet
            norm = mpl.colors.Normalize(vmin=self.Dbirth[0],
                                        vmax=self.Dbirth[-1])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        years = np.linspace(self.Dbirth.min(), self.Dbirth.max()+100, 1000)
        if not isinstance(plot_D14C, dict):
            plot_D14Cdict = {i: plot_D14C for i in variables}
        else:
            plot_D14Cdict = {i: True for i in variables}
            plot_D14Cdict.update(plot_D14C)
        for var in variables:
            axis = axies[var]
            if plot_D14Cdict[var]:
                axis.plot(years, self.model.Catm.lin(years), 'k:', zorder=6)
            for i in col_name_levels(self.solpd, 'sample'):
                axis.plot(self.Dbirth[i] + self.solpd.index,
                          self.solpd[var, i], '-',
                          c=sm.to_rgba(self.Dbirth[i]), zorder=1, lw=2)
            axis.set_title(var)
            axis.set_xlabel('Collection time')
            axis.set_ylabel(r'$\Delta 14C$')
            divider = make_axes_locatable(axis)
            cax = divider.append_axes(position='right', size='2%', pad=0.1)
            plt.colorbar(sm, ax=axis, cax=cax)
            cax.set_ylabel('Birth date')

        plt.tight_layout()
        return axies, sm

    def plot_delta(self, optimizer, parameters, axis=None, sm=None, cmap=None,
                   marker=None):
        if axis is None:
            axis = plt.gca()

        data = self.data.df

        if sm is None:
            if cmap is None:
                cmap = plt.cm.jet
            norm = mpl.colors.Normalize(vmin=data.Dcoll.min(),
                                        vmax=data.Dcoll.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

        divider = make_axes_locatable(axis)
        cax = divider.append_axes(position='right', size='2%', pad=0.1)
        axis.scatter(data.Dbirth,
                     optimizer.calc_sim_data_dict(parameters) - data.d14C,
                     c=data.Dcoll, edgecolor='black')
        axis.set_xlabel(r'Birth date')
        axis.set_ylabel(r'$\Delta14C_{{Model}} - \Delta14C_{{Data}} $')
        plt.colorbar(sm, ax=axis, cax=cax)
        return sm

    def plot_data_measurment(self, optimizer, parameters, axis=None, sm=None, cmap=None,
                   marker1='.' ,marker2='.',size1=20,size2=20,color=None):
        if axis is None:
            axis = plt.gca()

        data = self.data.df

        if sm is None:
            if cmap is None:
                cmap = plt.cm.jet
            norm = mpl.colors.Normalize(vmin=data.Dcoll.min(),
                                        vmax=data.Dcoll.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            divider = make_axes_locatable(axis)
            cax = divider.append_axes(position='right', size='2%', pad=0.1)
            plt.colorbar(sm, ax=axis, cax=cax)

        simd= optimizer.calc_sim_data_dict(parameters)
        #axis.scatter(data.Dbirth, data.d14C, c=sm.to_rgba(data.Dcoll), edgecolor='black',marker=marker1,zorder=3,s=size1)
        axis.scatter(data.Dbirth, data.d14C, c=sm.to_rgba(data.Dcoll), edgecolor=sm.to_rgba(data.Dcoll),marker=marker1,zorder=3,s=size1)
        axis.scatter(data.Dbirth,simd , color=sm.to_rgba(data.Dcoll), edgecolor=sm.to_rgba(data.Dcoll),marker=marker2,zorder=2,s=size2)
        for index,row in data.iterrows():
            axis.plot([row.Dbirth,row.Dbirth],[row.d14C,simd[index]],':', color=sm.to_rgba(row.Dcoll),alpha=1,zorder=1)
        axis.set_xlabel(r'Birth date')
        axis.set_ylabel(r'$\Delta14C$')


        return sm


    def plot_data(self, mode="coll", axis=None, sm=None, cmap=None,
                  marker=None):
        if axis is None:
            axis = plt.gca()
        if self.data is None:
            logging.info('no Data defined')
            return None
        data = self.data.df

        divider = make_axes_locatable(axis)
        cax = divider.append_axes(position='right', size='2%', pad=0.1)
        axis.set_title('C14 Data')

        if mode == "coll":
            if sm is None:
                if cmap is None:
                    cmap = plt.cm.jet
                norm = mpl.colors.Normalize(vmin=data.Dbirth.min(),
                                            vmax=data.Dbirth.max())
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
            axis.scatter(data.Dcoll, data.d14C, c=sm.to_rgba(data.Dbirth),
                         edgecolor='black', zorder=10, marker=marker)
            axis.set_xlabel('Collection time')
            axis.set_ylabel(r'$\Delta 14C$ as measured')
            plt.colorbar(sm, ax=axis, cax=cax)
            cax.set_ylabel('Birth date')
        elif mode == 'birth':
            if sm is None:
                if cmap is None:
                    cmap = plt.cm.jet
                norm = mpl.colors.Normalize(vmin=data.Dcoll.min(),
                                            vmax=data.Dcoll.max())
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
            axis.scatter(data.Dbirth, data.d14C, c=sm.to_rgba(data.Dcoll),
                         edgecolor='black', zorder=10, marker=marker)
            axis.set_xlabel('Birth date')
            axis.set_ylabel(r'$\Delta 14C$ as measured')
            norm.vmin = data.Dcoll.min()
            norm.vmax = data.Dcoll.max()
            plt.colorbar(sm, ax=axis, cax=cax)
            cax.set_ylabel('Collection time')
        years = np.linspace(self.Dbirth.min(), self.Dbirth.max()+self.age,
                            1000)
        axis.plot(years, self.model.Catm.lin(years), 'k:', zorder=6)
        return axis, cax, sm

    def plot_simdata(self, axis=None, sm=None, cmap=None, marker=None,
                     insert=True, insert_loc=1, insert_width="40%",
                     insert_height="50%"):
        if axis is None:
            axis = plt.gca()
        if self.data is None:
            logging.info('no Data defined')
            return None
        data = self.data.df

        try:
            self.solpd_data
        except AttributeError:
            logging.info('run odeint first')
            return None
        if sm is None:
            if cmap is None:
                cmap = plt.cm.jet
            norm = mpl.colors.Normalize(vmin=data.Dbirth.min(),
                                        vmax=data.Dbirth.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

        axis.scatter(data.Dcoll, data.d14C, c=sm.to_rgba(data.Dbirth),
                     edgecolor='black', zorder=10, marker=marker)
        axis.errorbar(data.Dcoll, data.d14C, yerr=data.e14C, fmt='.')
        years = np.linspace(self.Dbirth.min(), self.Dbirth.max()+self.age,
                            1000)
        axis.plot(years, self.model.Catm.lin(years), 'k:', zorder=6)

        measure_sim = []
        for idx, row in self.solpd_data.iterrows():
            measure_sim.append(self.model.measurement_model(row, self.data))
        measure_sim = pd.DataFrame(measure_sim)
        for idx, col in measure_sim.iteritems():
            axis.plot(data.Dbirth[idx] + measure_sim.index, col, '-',
                      c=sm.to_rgba(data.Dbirth[idx]), zorder=1, lw=2)
        axis.set_xlabel('Collection time')
        axis.set_ylabel(r'$\Delta 14C$ as measured')
        divider = make_axes_locatable(axis)
        cax = divider.append_axes(position='right', size='2%', pad=0.1)
        plt.colorbar(sm, ax=axis, cax=cax)
        cax.set_ylabel('Birth date')
        if insert:
            iaxes = inset_axes(axis, loc=insert_loc, width=insert_width,
                               height=insert_height)
            iaxes.scatter(data.Dcoll, data.d14C, c=sm.to_rgba(data.Dbirth),
                          edgecolor='black', zorder=10, marker=marker)
            iaxes.errorbar(data.Dcoll, data.d14C, yerr=data.e14C, fmt='.')

            iaxes.plot(years, self.model.Catm.lin(years), 'k:', zorder=6)

            for idx, col in measure_sim.iteritems():
                iaxes.plot(data.Dbirth[idx] + measure_sim.index, col, '-',
                           c=sm.to_rgba(data.Dbirth[idx]), zorder=1, lw=2)
            iaxes.set_xticks([])
            iaxes.set_yticks([])
            iaxes.set_xlim(data.Dcoll.min()-1, data.Dcoll.max()+1)
            iaxes.set_ylim(data.d14C.min()-0.05, data.d14C.max()+0.05)
            mark_inset(axis, iaxes, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.draw()
        return sm

    def plot_sim_birth(self, mapfunction=None,collection_years=None, axis=None, sm=None,
                           cmap=None,make_colorbar=True):
        '''
        WARNING only works if samples for gerneric data are  np.arange(birthdate_min, birthdate_max) 
        '''
        if axis is None:
            axis = plt.gca()
        if self.data is None:
            logging.info('no Data defined')
            return None
        data = self.data.df
        try:
            self.solpd
        except AttributeError:
            logging.info('run odeint first')
            return None

        if sm is None:
            if cmap is None:
                cmap = plt.cm.jet
            norm = mpl.colors.Normalize(vmin=data.Dcoll.min(),
                                        vmax=data.Dcoll.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        else:
            make_colorbar=False
        if collection_years is None:
            collection_years = [data.Dcoll.mean()]

        if mapfunction is None:
            mapfunction = lambda i,x: x.loc[variable,:]

        measure_sim = []
        for idx, row in self.solpd.iterrows():
            measure_sim.append(mapfunction(idx,row))
        measure_sim = pd.DataFrame(measure_sim)
        for collyear in collection_years:
            age = collyear - self.Dbirth
            dfa = measure_sim.reindex(age, method='nearest')
            mask = np.logical_and(age<self.age,self.Dbirth<=collyear)
            axis.plot(self.Dbirth[mask], np.diag(dfa)[mask], '-', c=sm.to_rgba(collyear),
                      zorder=1, lw=2)
        axis.set_xlabel('Birth date')
        axis.set_ylabel(r'$\Delta 14C$ as measured')
        if make_colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes(position='right', size='2%', pad=0.1)
            plt.colorbar(sm, ax=axis, cax=cax)
            cax.set_ylabel('Collection date')

        return sm

    def plot_chain(self, chain, pin=None, fnum=None):
        f = plt.figure(fnum, figsize=(10, 5*self.model.nparas))
        shape = (self.model.nparas, 1)
        for j in range(self.model.nparas):
            ax = plt.subplot2grid(shape, (j, 0), fig=f)
            ax.set_title(str(self.model.parameter_names[j]))
            i = 0
            if pin is not None:
                leng = chain.shape[1] - 1
                p = pin[self.model.parameter_names[j]]
                ax.plot([0, leng], [p, p], lw=2, c='black')
            for _ in range(100):
                try:
                    ax.plot(chain[i, :, j])
                    i = i+1
                except IndexError:
                    break
        return f
