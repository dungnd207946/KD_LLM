import pandas as pd
import torch
from torch import nn
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

class MultiTeacherDataCollator(DataCollatorForSeq2Seq):
    """
    A custom data collator that prepares batch data for training with multiple teacher models.
    Extends the DataCollatorForSeq2Seq to handle additional inputs for each teacher model.
    """
    def __call__(self, features, return_tensors=None):
        # Convert list of features to DataFrame for easy manipulation
        features_df = pd.DataFrame(features)

        # Separate features for the prediction model and each teacher model
        pred_features = features_df.loc[:, ~features_df.columns.isin(['t5_labels', 't5_input_ids', 't5_attention_mask', 'llama_labels', 'llama_input_ids', 'llama_attention_mask'])].to_dict('records')
        t5_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask', 'llama_labels' , 'llama_input_ids', 'llama_attention_mask'])].rename(
            columns={'t5_labels': 'labels', 't5_input_ids': 'input_ids', 't5_attention_mask': 'attention_mask'}).to_dict('records')
        llama_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask', 't5_labels', 't5_input_ids', 't5_attention_mask'])].rename(
            columns={'llama_labels': 'labels', 'llama_input_ids': 'input_ids', 'llama_attention_mask': 'attention_mask'}).to_dict('records')

        # Prepare each set of features using the parent class's method
        pred_features = super().__call__(pred_features, return_tensors)
        t5_features = super().__call__(t5_features, return_tensors)
        llama_features = super().__call__(llama_features, return_tensors)

        # Return a dictionary containing prepared features for all models
        return {'pred': pred_features, 't5': t5_features, 'llama': llama_features}

class MultiTeacherTrainer(Seq2SeqTrainer):
    """
    A custom trainer for sequence-to-sequence models that incorporates multiple teacher models.
    Extends the Seq2SeqTrainer to compute loss based on the outputs of the prediction model and teacher models.
    """
    def __init__(self, gamma, alpha, beta, output_rationale, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma  # Weight for the prediction model's loss
        self.alpha = alpha  # Weight for the first teacher model's loss
        self.beta = beta    # Weight for the second teacher model's loss
        self.output_rationale = output_rationale  # Whether to output rationales

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute outputs and loss for each model
        pred_outputs = model(**inputs['pred'])
        t5_outputs = model(**inputs['t5'])
        llama_outputs = model(**inputs['llama'])

        # Compute combined loss using specified weights
        loss = self.gamma * pred_outputs.loss + self.alpha * t5_outputs.loss + self.beta * llama_outputs.loss

        # Return loss and optionally the model outputs
        return (loss, {'pred': pred_outputs, 't5': t5_outputs, 'llama': llama_outputs}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Override prediction step to handle multiple models
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        if self.output_rationale:
            t5_outputs = super().prediction_step(model, inputs['t5'], prediction_loss_only=False, ignore_keys=ignore_keys)
            llama_outputs = super().prediction_step(model, inputs['llama'], prediction_loss_only=False, ignore_keys=ignore_keys)
        else:
            t5_outputs = llama_outputs = pred_outputs

        # Compute combined lossfrom each model's outputs and return combined loss and outputs.

        # Extract loss values from outputs
        pred_outputs_loss = pred_outputs[0] if isinstance(pred_outputs, tuple) else pred_outputs.loss
        t5_outputs_loss = t5_outputs[0] if isinstance(t5_outputs, tuple) else t5_outputs.loss
        llama_outputs_loss = llama_outputs[0] if isinstance(llama_outputs, tuple) else llama_outputs.loss

        # Compute total loss
        loss = self.gamma * pred_outputs_loss + self.alpha * t5_outputs_loss + self.beta * llama_outputs_loss

        # Return total loss and individual model outputs
        return loss, [pred_outputs[1], t5_outputs[1], llama_outputs[1]], [pred_outputs[2], t5_outputs[2], llama_outputs[2]]