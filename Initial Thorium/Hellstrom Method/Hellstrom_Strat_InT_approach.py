
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

class U_Series_Age_Equation:
    def __init__(self, r08, r08_err,
                 r28, r28_err,
                 r48, r48_err,
                 r02_initial,
                 r02_initial_err,
                 r48_detrital,
                 r48_detrital_err,
                 r28_detrital,
                 r28_detrital_err,
                 rho_28_48 = 0.0,
                 rho_08_48 = 0.0,
                 rho_08_28 = 0.0,
                 r08_detrital = 0.0,
                 r08_detrital_err = 0.0):
        
        self.r08 = r08
        self.r08_err = r08_err
        self.r28 = r28
        self.r28_err = r28_err
        self.r48 = r48
        self.r48_err = r48_err
        self.r02_initial = r02_initial
        self.r02_initial_err = r02_initial_err
        self.r48_detrital = r48_detrital
        self.r48_detrital_err = r48_detrital_err
        self.r28_detrital = r28_detrital
        self.r28_detrital_err = r28_detrital_err
        self.lambda_230 = 9.1577e-6
        self.lambda_234 = 2.8263e-6
        self.lambda_230_err = 1.3914e-8
        self.lambda_234_err = 2.8234e-9
        self.rho_28_48 = rho_28_48
        self.rho_08_48 = rho_08_48
        self.rho_08_28 = rho_08_28
        self.r08_detrital = r08_detrital
        self.r08_detrital_err = r08_detrital_err


    def Age_Equation(self, T):
    
        A = self.r08 - self.r28 * self.r02_initial * np.exp(-self.lambda_230 * T)
        B = 1 - np.exp(-self.lambda_230 * T)
        D = self.r48 - 1
        E = self.lambda_230 / (self.lambda_234 - self.lambda_230)
        F = 1 - np.exp((self.lambda_234 - self.lambda_230)*T)
        C = D * E * F
        return A - B + C



    def Age_solver(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]

    def Age_solver_Ludwig(self, age_guess=1e4):
        """
        Solve the U-series age equation for a single age estimate.
        """
        func = lambda age: self.Age_Equation_w_Ludwig(age)
        t_solution = fsolve(func, age_guess)
        return t_solution[0]


    def Ages_And_Age_Uncertainty_Calculation_w_InitialTh(self):
        """
        Age and uncertainty calculation
        decay constant uncertainties not included here
        """
        Age = self.Age_solver()
        
        # Compute lambda difference
        lam_diff = self.lambda_234 - self.lambda_230
        
        # Compute df/dT components
        df_dT_1 = self.lambda_230 * self.r28 * self.r02_initial * np.exp(-self.lambda_230 * Age)
        df_dT_2 = -self.lambda_230 * np.exp(-self.lambda_230 * Age)
        df_dT_3 = - (self.r48 - 1) * self.lambda_230 * np.exp(lam_diff * Age)
        df_dT = df_dT_1 + df_dT_2 + df_dT_3
        
        # Compute partial derivatives dt/dx_i
        dt_dr08 = -1 / df_dT
        dt_dr28 = (self.r02_initial * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr02 = (self.r28 * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr48 = - ((self.lambda_230 / lam_diff) * (1 - np.exp(lam_diff * Age))) / df_dT
    
        age_jacobian = np.array([dt_dr08,
                                 dt_dr28,
                                 dt_dr02,
                                 dt_dr48])
    
        cov_age = np.zeros((4,4))
        cov_age[0,0] = self.r08_err**2
        cov_age[0,1] = cov_age[1,0] = self.rho_08_28 * self.r08_err * self.r28_err
        cov_age[1,1] = self.r28_err**2
        cov_age[2,2] = self.r02_initial_err**2
        cov_age[3,3] = self.r48_err**2
        cov_age[1,3] = cov_age[3,1] = self.rho_28_48 * self.r48_err * self.r28_err
        cov_age[0,3] = cov_age[3,0] = self.rho_08_48 * self.r48_err * self.r08_err
    
        age_err = np.dot(age_jacobian, np.dot(cov_age, age_jacobian))
    
        return Age, np.sqrt(age_err)


class Hellstrom_appraoch:
    """
    Function to estimate a prior for the initial 230Th
    this will incorporate uncertainty into the measurement and provide a robust series of initial
    230Th estimations
    - We can then combined these into a distribution for the sample
    - Will do a bootstrap resampled to get distribution
    """
    def __init__(self, r08, r08_err,
                       r28, r28_err,
                       r48, r48_err):

        # Helper things
        self.Th230_lam_Cheng = 9.1577e-06
        self.Th230_lam_Cheng_err = 1.3914e-08
        self.U234_lam_Cheng = 2.8263e-06
        self.U234_lam_Cheng_err = 2.8234e-09
        self.N_ratios = r08.shape[0]
        self.n_samples = 10000
        
        
        # Unpack measured activity ratios
        # These are assumed to be 1-sigma here
        self.r08 = r08
        self.r28 = r28
        self.r48 = r48
        self.r08_err= r08_err
        self.r28_err = r28_err
        self.r48_err = r48_err
        
        # Empty tuples for later
        self.r02_initial = None
        self.r02_initial_err = None
        self.r02_initial_mean = None
        self.r02_initial_std = None
        

    def check_stratigraphic_order_with_tolerance(self, ages, age_errs, tolerance=1):
        """
        Returns True if the fraction of age intervals that are in correct stratigraphic
        order is >= `tolerance`. Otherwise returns False.
        
        If `tolerance=1.0`, it reverts to the original strict behavior (i.e.,
        all intervals must be ordered).
        """
        diff_ages = np.diff(ages)
        err_diff = np.sqrt(age_errs[:-1]**2 + age_errs[1:]**2)
        
        n_intervals = len(diff_ages)
        n_in_order = 0
        
        for i in range(n_intervals):
            # Condition to check if we consider the i-th interval "in order"
            # The usual check is:
            #   If diff_ages[i] < 0 by more than the combined error, it's out of order.
            #   Otherwise, it's considered in order (within error).
            # So 'out of order' means: diff_ages[i] < -err_diff[i]
            
            if not (diff_ages[i] < 0 and abs(diff_ages[i]) > err_diff[i]):
                n_in_order += 1

        fraction_in_order = n_in_order / n_intervals
        return fraction_in_order >= tolerance
        
        
  
    def Ages_And_Age_Uncertainty_Calculation_w_InitialTh(self):
        """
        Age and uncertainty calculation
        decay constant uncertainties not included here
        """
        Age = self.Age_solver()
        
        # Compute lambda difference
        lam_diff = self.lambda_234 - self.lambda_230
        
        # Compute df/dT components
        df_dT_1 = self.lambda_230 * self.r28 * self.r02_initial * np.exp(-self.lambda_230 * Age)
        df_dT_2 = -self.lambda_230 * np.exp(-self.lambda_230 * Age)
        df_dT_3 = - (self.r48 - 1) * self.lambda_230 * np.exp(lam_diff * Age)
        df_dT = df_dT_1 + df_dT_2 + df_dT_3
        
        # Compute partial derivatives dt/dx_i
        dt_dr08 = -1 / df_dT
        dt_dr28 = (self.r02_initial * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr02 = (self.r28 * np.exp(-self.lambda_230 * Age)) / df_dT
        dt_dr48 = - ((self.lambda_230 / lam_diff) * (1 - np.exp(lam_diff * Age))) / df_dT
    
        age_jacobian = np.array([dt_dr08,
                                 dt_dr28,
                                 dt_dr02,
                                 dt_dr48])
    
        cov_age = np.zeros((4,4))
        cov_age[0,0] = self.r08_err**2
        cov_age[0,1] = cov_age[1,0] = self.rho_08_28 * self.r08_err * self.r28_err
        cov_age[1,1] = self.r28_err**2
        cov_age[2,2] = self.r02_initial_err**2
        cov_age[3,3] = self.r48**2
        cov_age[1,3] = cov_age[3,1] = self.rho_28_48 * self.r48_err * self.r28_err
        cov_age[0,3] = cov_age[3,0] = self.rho_08_48 * self.r48_err * self.r08_err
    
        age_err = np.dot(age_jacobian, np.dot(cov_age, age_jacobian))
    
        return Age, np.sqrt(age_err)
        
        
    def uniform_log_sample(self, lower=0.01, upper=300):
        # Sample uniformly in log space:
        u = np.random.uniform(0, 1)
        candidate = lower * (upper/lower)**u
        return candidate

        
    def Get_Initial_Thoriums(self,
                            min_valid=10000,         # Minimum number of accepted candidates required
                            batch_size=50000,       # Number of candidates to sample per batch
                            max_attempts=100000,    # Maximum total candidates to try
                            tolerance=1.00,         # Initial stratigraphic order tolerance (e.g., 100% intervals in order)
                            attempts_before_relax=5000,  # Attempts before relaxing tolerance
                            relaxation_factor=0.05,      # How much to relax the tolerance (i.e., lower it)
                            max_negative_ages=0):
        """
        Vectorized/multi-batch sampling of initial 230Th values and uncertainties that yield
        a sufficient fraction of age intervals in stratigraphic order.
        Instead of a continuous penalty, a binary test is performed using a tolerance.
        """
        accepted_r02 = []
        accepted_r02_err = []
        accepted_ages = []
        accepted_age_errs = []
        accepted_tolerances = []
        attempts_since_last_valid = 0
        total_attempts = 0
        current_tolerance = tolerance
    
        while len(accepted_r02) < min_valid and total_attempts < max_attempts:
            candidates = np.array([self.uniform_log_sample(lower=0.01, upper=150)
                                for _ in range(batch_size)])
            # For uncertainties, sample a fractional error uniformly
            candidate_errs = (np.array([np.random.uniform(0.05, 0.5)
                                        for _ in range(batch_size)])
                            * candidates)
    
            ages_all = np.zeros((batch_size, self.N_ratios))
            age_errs_all = np.zeros((batch_size, self.N_ratios))
            valid_mask = np.ones(batch_size, dtype=bool)
    
            for idx in range(self.N_ratios):
                for j in range(batch_size):
                    try:
                        U_age = U_Series_Age_Equation(
                            self.r08[idx], self.r08_err[idx],
                            self.r28[idx], self.r28_err[idx],
                            self.r48[idx], self.r48_err[idx],
                            candidates[j], candidate_errs[j],
                            0, 0, 0, 0)
                        age, age_err = U_age.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
                        ages_all[j, idx] = age
                        age_errs_all[j, idx] = age_err
                    except Exception as e:
                        ages_all[j, idx] = np.nan
                        age_errs_all[j, idx] = np.nan
                        valid_mask[j] = False
    
            valid_mask &= ~np.any(np.isnan(ages_all), axis=1)
            valid_mask &= (np.sum(ages_all < 0, axis=1) <= max_negative_ages)
            valid_mask &= ~np.any(ages_all + 2 * age_errs_all < 0, axis=1)
    
            batch_valid_indices = []
            # Loop over only valid indices in this batch.
            for j in np.where(valid_mask)[0]:
                # Check if the candidate meets the stratigraphic ordering tolerance.
                # Accept if the fraction of in-order intervals is >= current_tolerance.
                if self.check_stratigraphic_order_with_tolerance(ages_all[j, :],
                                                                age_errs_all[j, :],
                                                                tolerance=current_tolerance):
                    batch_valid_indices.append(j)
                    # Early exit: if total accepted meets/exceeds the target, break.
                    if len(accepted_r02) + len(batch_valid_indices) >= min_valid:
                        break
    
            total_attempts += batch_size
    
            if len(batch_valid_indices) == 0:
                attempts_since_last_valid += batch_size
            else:
                attempts_since_last_valid = 0
    
            if attempts_since_last_valid >= attempts_before_relax:
                # Relax tolerance: lower the fraction required.
                current_tolerance = max(0, current_tolerance - relaxation_factor)
                attempts_since_last_valid = 0
                print(f"Relaxing stratigraphic order tolerance to {current_tolerance:.3f} after {total_attempts} attempts.")
    
            if len(batch_valid_indices) > 0:
                accepted_r02.extend(candidates[batch_valid_indices])
                accepted_r02_err.extend(candidate_errs[batch_valid_indices])
                accepted_ages.extend(ages_all[batch_valid_indices])
                accepted_age_errs.extend(age_errs_all[batch_valid_indices])
                accepted_tolerances.extend([current_tolerance] * len(batch_valid_indices))
                print(f"Batch at attempt {total_attempts}: Found {len(batch_valid_indices)} valid candidates  (total so far: {len(accepted_r02)}).")
    
            # Break early if we have enough samples.
            if len(accepted_r02) >= min_valid:
                break
    
        # After the loop, convert lists to arrays.
        accepted_r02 = np.array(accepted_r02)
        accepted_r02_err = np.array(accepted_r02_err)
        accepted_ages = np.array(accepted_ages)
        accepted_age_errs = np.array(accepted_age_errs)
        accepted_tolerances = np.array(accepted_tolerances)
    
        if len(accepted_r02) < min_valid:
            print(f"Warning: Only found {len(accepted_r02)} valid candidates after {total_attempts} attempts.")
        else:
            # Ensure that self.n_samples is defined, or pass a parameter instead.
            indices = np.random.choice(len(accepted_r02), self.n_samples, replace=True)
            accepted_r02 = accepted_r02[indices]
            accepted_r02_err = accepted_r02_err[indices]
            accepted_ages = accepted_ages[indices]
            accepted_age_errs = accepted_age_errs[indices]
            accepted_tolerances = accepted_tolerances[indices]
    
        return accepted_r02, accepted_r02_err, accepted_ages, accepted_age_errs, accepted_tolerances


    

    def Get_Init_Thor_samples(self):
        self.r02_initial, self.r02_initial_err, _, _, self.scores, = self.Get_Initial_Thoriums()
        
    def Plot_samples(self, save = False):
        if self.r02_initial is None:
            self.Get_Init_Thor_samples()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize = (5,5))
        
        plt.plot(self.r02_initial, (self.r02_initial_err/self.r02_initial)*200, '+',
                markersize = 8,color = 'slategrey',
                label = 'Hellstrom, (2006)\nType Method')
        
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Simulated mean $^{230}$Th/$^{232}$Th$_{0}$')
        plt.ylabel('Simulated $^{230}$Th/$^{232}$Th uncertainty (%) (2$\sigma$)')
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig('Initial_Thor_test.png', bbox_inches = 'tight', dpi = 300)
        
    def Get_Summary_Values(self):
        # A way of bootstrapping the best
        # estimate for the global initial thorium distribution
        if self.r02_initial is None:
            self.Get_Init_Thor_samples()
        num = 30000
        samples = []
        for i in range(num):
            sample_mc = np.random.normal(self.r02_initial, self.r02_initial_err)
            
            samples.append(sample_mc)
            
        
        sample_array = np.array(samples)
        
        self.r02_initial_mean = sample_array.flatten().mean()
        self.r02_initial_std = sample_array.flatten().std()
        
        print(f"230Th/232Th$_0$ = {self.r02_initial_mean:.2f} ± {self.r02_initial_std:.2f} (Activity Ratio)")
        

            
            

    def Use_Correction_for_Ages(self):
        ages_all = np.zeros(self.N_ratios)
        age_errs_all = np.zeros(self.N_ratios)

        for idx in range(self.N_ratios):

            U_age = U_Series_Age_Equation(
                self.r08[idx], self.r08_err[idx],
                self.r28[idx], self.r28_err[idx],
                self.r48[idx], self.r48_err[idx],
                self.r02_initial_mean, self.r02_initial_std,
                0, 0, 0, 0)
            age, age_err = U_age.Ages_And_Age_Uncertainty_Calculation_w_InitialTh()
            ages_all[idx] = age
            age_errs_all[idx] = age_err

        return ages_all, age_errs_all
