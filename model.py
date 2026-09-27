import numpy as np
from sklearn.ensemble import RandomForestRegressor

class SolarPowerPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.nominal_voltage = 6.0    # 6V mini solar panel
        self.nominal_max_power = 2.0  # 2 Watts clean max rating
        self._train_baseline_model()

    def _train_baseline_model(self):
        """Trains baseline clean behavior: accounts for light and temp derating."""
        np.random.seed(42)
        n_samples = 1500

        lux_samples = np.random.uniform(0, 1200, n_samples)
        temp_samples = np.random.uniform(15, 50, n_samples)

        # Standard crystalline silicon loss: 0.4% per °C over 25°C
        temp_derate = np.maximum(0.0, temp_samples - 25.0) * 0.004
        clean_power = (lux_samples / 1000.0) * self.nominal_max_power * (1.0 - temp_derate)
        clean_power = np.clip(clean_power, 0.0, None)

        noise = np.random.normal(0, 0.015, n_samples)
        y = np.clip(clean_power + noise, 0.0, None)

        X = np.column_stack((lux_samples, temp_samples))
        self.model.fit(X, y)

    def predict_clean_power(self, lux: float, temp: float) -> float:
        if lux < 30.0:
            return 0.0
        return max(0.0, float(self.model.predict([[lux, temp]])[0]))

    def diagnose(self, lux: float, temp: float, voltage: float, current_ma: float):
        p_actual = max(0.0, voltage * (current_ma / 1000.0))
        p_predicted = self.predict_clean_power(lux, temp)

        # 1. Night or low irradiance standby
        if lux < 40.0:
            return {
                "status": "NIGHT_IDLE",
                "badge": "Night / Standby",
                "color": "slate",
                "p_actual": round(p_actual, 3),
                "p_predicted": 0.0,
                "loss_percentage": 0.0,
                "efficiency": 0.0
            }

        # Loss ratio calculation
        if p_predicted <= 0.01:
            loss_pct = 0.0
            efficiency = 100.0
        else:
            loss_pct = max(0.0, min(100.0, ((p_predicted - p_actual) / p_predicted) * 100.0))
            efficiency = max(0.0, min(100.0, (p_actual / p_predicted) * 100.0))

        # 2. Hardware Fault: High loss + voltage collapses (<75% nominal)
        if loss_pct >= 15.0 and voltage < (0.75 * self.nominal_voltage):
            status = "HARDWARE_FAULT"
            badge = "Hardware Fault - Check Cells/Diodes"
            color = "amber"

        # 3. Dust / Soiling: High loss while voltage stays normal (>=80% nominal)
        elif loss_pct >= 15.0 and voltage >= (0.80 * self.nominal_voltage):
            status = "DUST_DETECTED"
            badge = "Dust Detected - Cleaning Required!"
            color = "red"

        # 4. Normal Clean
        else:
            status = "NORMAL_CLEAN"
            badge = "Panel Clean - Optimal Efficiency"
            color = "green"

        return {
            "status": status,
            "badge": badge,
            "color": color,
            "p_actual": round(p_actual, 3),
            "p_predicted": round(p_predicted, 3),
            "loss_percentage": round(loss_pct, 1),
            "efficiency": round(efficiency, 1)
        }

predictor = SolarPowerPredictor()