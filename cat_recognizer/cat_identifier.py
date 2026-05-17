"""Load a trained cat model and classify new images."""
import joblib
import numpy as np

from feature_extractor import extract_features, l2_normalize


class CatIdentifier:
    def __init__(self, model_path, unknown_threshold=0.6, use_knn=True, k=3):
        obj = joblib.load(model_path)
        self.classes = obj['classes']
        self.prototypes = obj['prototypes']
        self.samples = obj.get('samples')
        self.sample_labels = obj.get('sample_labels')
        self.unknown_threshold = unknown_threshold
        self.use_knn = use_knn and self.samples is not None
        self.k = k

    def identify(self, img_path):
        v = l2_normalize(extract_features(img_path))

        proto_sims = self.prototypes @ v
        best_idx = int(np.argmax(proto_sims))
        best_score = float(proto_sims[best_idx])
        label = self.classes[best_idx]

        if self.use_knn:
            sample_sims = self.samples @ v
            top = np.argsort(-sample_sims)[: self.k]
            votes = {}
            for idx in top:
                lab = self.sample_labels[idx]
                votes[lab] = votes.get(lab, 0.0) + float(sample_sims[idx])
            label = max(votes, key=votes.get)
            best_score = max(best_score, float(sample_sims[top[0]]))

        if best_score < self.unknown_threshold:
            return 'unknown', best_score
        return label, best_score
