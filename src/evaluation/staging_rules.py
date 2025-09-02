import numpy as np


class TumorStaging:
    """
    Medical staging rules for brain tumors based on WHO guidelines
    """

    def __init__(self):
        self.staging_criteria = {
            'glioma': self._stage_glioma,
            'meningioma': self._stage_meningioma,
            'pituitary': self._stage_pituitary
        }

    def stage_tumor(self, tumor_type, volume_mm3, shape_features,
                    max_diameter_mm, location=None):
        """
        Determine tumor stage based on type and characteristics
        """
        if tumor_type in self.staging_criteria:
            return self.staging_criteria[tumor_type](
                volume_mm3, shape_features, max_diameter_mm, location
            )
        else:
            return self._general_staging(volume_mm3, shape_features, max_diameter_mm)

    def _stage_glioma(self, volume_mm3, shape_features, max_diameter_mm, location):
        """
        WHO Grade I-IV staging for gliomas
        """
        stage_info = {
            'grade': None,
            'description': '',
            'prognosis': '',
            'malignancy': 'benign'
        }

        # Size-based initial grading
        if max_diameter_mm < 20:
            grade = 'I'
            description = 'Pilocytic astrocytoma (likely)'
            prognosis = 'Excellent with complete resection'
            malignancy = 'benign'
        elif max_diameter_mm < 40:
            grade = 'II'
            description = 'Diffuse astrocytoma'
            prognosis = 'Good with treatment (5-7 years median survival)'
            malignancy = 'low-grade'
        elif max_diameter_mm < 60:
            grade = 'III'
            description = 'Anaplastic astrocytoma'
            prognosis = 'Moderate (2-3 years median survival)'
            malignancy = 'malignant'
        else:
            grade = 'IV'
            description = 'Glioblastoma multiforme'
            prognosis = 'Poor (12-15 months median survival)'
            malignancy = 'highly_malignant'

        # Adjust based on shape irregularity
        if shape_features['irregularity'] > 0.8 and grade in ['I', 'II']:
            grade = 'III'
            description = 'High irregularity suggests anaplastic features'
            malignancy = 'malignant'

        stage_info['grade'] = f'WHO Grade {grade}'
        stage_info['description'] = description
        stage_info['prognosis'] = prognosis
        stage_info['malignancy'] = malignancy

        return stage_info

    def _stage_meningioma(self, volume_mm3, shape_features, max_diameter_mm, location):
        """
        WHO Grade I-III staging for meningiomas
        """
        stage_info = {
            'grade': None,
            'description': '',
            'prognosis': '',
            'malignancy': 'benign'
        }

        # Most meningiomas are benign
        if shape_features['circularity'] > 0.7 and max_diameter_mm < 50:
            grade = 'I'
            description = 'Benign meningioma'
            prognosis = 'Excellent (>90% 5-year survival)'
            malignancy = 'benign'
        elif shape_features['irregularity'] > 0.5 or max_diameter_mm > 50:
            grade = 'II'
            description = 'Atypical meningioma'
            prognosis = 'Good (70-80% 5-year survival)'
            malignancy = 'intermediate'
        else:
            grade = 'III'
            description = 'Anaplastic/malignant meningioma'
            prognosis = 'Poor (55% 5-year survival)'
            malignancy = 'malignant'

        stage_info['grade'] = f'WHO Grade {grade}'
        stage_info['description'] = description
        stage_info['prognosis'] = prognosis
        stage_info['malignancy'] = malignancy

        return stage_info

    def _stage_pituitary(self, volume_mm3, shape_features, max_diameter_mm, location):
        """
        Classification for pituitary adenomas
        """
        stage_info = {
            'grade': None,
            'description': '',
            'prognosis': '',
            'malignancy': 'benign'
        }

        if max_diameter_mm < 10:
            classification = 'Microadenoma'
            description = 'Small pituitary adenoma'
            prognosis = 'Excellent with treatment'
            malignancy = 'benign'
        elif max_diameter_mm < 40:
            classification = 'Macroadenoma'
            description = 'Large pituitary adenoma'
            prognosis = 'Good with surgery and medication'
            malignancy = 'benign'
        else:
            classification = 'Giant adenoma'
            description = 'Very large pituitary tumor'
            prognosis = 'Variable, depends on invasion'
            malignancy = 'locally_aggressive'

        stage_info['grade'] = classification
        stage_info['description'] = description
        stage_info['prognosis'] = prognosis
        stage_info['malignancy'] = malignancy

        return stage_info

    def _general_staging(self, volume_mm3, shape_features, max_diameter_mm):
        """
        General staging for unspecified tumor types
        """
        stage_info = {
            'grade': 'Unknown',
            'description': 'Tumor type not specified',
            'prognosis': 'Requires further evaluation',
            'malignancy': 'undetermined'
        }

        # Basic size-based classification
        if max_diameter_mm < 30 and shape_features['circularity'] > 0.7:
            stage_info['malignancy'] = 'likely_benign'
        elif max_diameter_mm > 50 or shape_features['irregularity'] > 0.7:
            stage_info['malignancy'] = 'likely_malignant'
        else:
            stage_info['malignancy'] = 'intermediate'

        return stage_info

    def get_treatment_recommendation(self, stage_info, tumor_type):
        """
        Provide treatment recommendations based on staging
        """
        recommendations = []

        if stage_info['malignancy'] in ['benign', 'likely_benign']:
            recommendations.append('Surgical resection recommended if symptomatic')
            recommendations.append('Regular monitoring with MRI every 6-12 months')
        elif stage_info['malignancy'] in ['low-grade', 'intermediate']:
            recommendations.append('Surgical resection followed by observation')
            recommendations.append('Consider radiation therapy if incomplete resection')
            recommendations.append('MRI follow-up every 3-6 months')
        else:  # malignant
            recommendations.append('Urgent neurosurgical consultation')
            recommendations.append('Maximal safe resection')
            recommendations.append('Concurrent chemoradiation therapy')
            recommendations.append('Consider clinical trials')

        return recommendations