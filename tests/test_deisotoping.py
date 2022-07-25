import os
import pickle as pkl
import numpy as np

from mass_automation.deisotoping.process import LinearDeisotoper, MlDeisotoper

os.environ.get('ML_MODEL_PATH')
linear_model_name = 'linear_model.pkl'
ml_model_names = ['rf_model.pkl', 'xgb_model.pkl', 'cb_model.pkl']
sonogashira_filenames = ['react_mix1.pkl', 'react_mix2.pkl', 'react_mix3.pkl']


class TestLinearDeisotoper:
    def test_tea_samples(self):
        root = os.path.abspath('.')
        path = os.path.join(root, 'data', 'mass_spectra', 'tea_spec.pkl')
        with open(path, 'rb') as f:
            spectrum = pkl.load(f)

        linear_model_path = os.path.join(root, 'data', 'models', linear_model_name)
        deisotoper = LinearDeisotoper().load(linear_model_path)
        predictions = deisotoper(spectrum)
        assert len(np.unique(predictions)) >= 40

    def test_sonogashira_samples(self):
        for filename in sonogashira_filenames:
            root = os.path.abspath('.')
            path = os.path.join(root, 'data', 'mass_spectra', filename)
            with open(path, 'rb') as f:
                spectrum = pkl.load(f)

            linear_model_path = os.path.join(root, 'data', 'models', linear_model_name)
            deisotoper = LinearDeisotoper().load(linear_model_path)
            predictions = deisotoper(spectrum)
            assert len(np.unique(predictions)) >= 40


class TestMlDeisotoper:
    def test_sonogashira_samples(self):
        s = 0
        for model_name in ml_model_names:
            for filename in sonogashira_filenames:
                s+=1
                if s < 8:
                    root = os.path.abspath('.')
                    path = os.path.join(root, 'data', 'mass_spectra', filename)
                    with open(path, 'rb') as f:
                        spectrum = pkl.load(f)

                    model_path = os.path.join(root, 'data', 'models', model_name)
                    deisotoper = MlDeisotoper().load(model_path)
                    predictions = deisotoper(spectrum)
                    assert len(np.unique(predictions)) >= 33
                else:
                    break