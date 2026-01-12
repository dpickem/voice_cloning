#!/usr/bin/env python3
"""Pydantic models used across the voice cloning server."""

from __future__ import annotations

from pydantic import BaseModel


class Sample(BaseModel):
    """A single training or evaluation sample."""

    audio_file: str
    text: str
    speaker_name: str
    language: str


class DataConfig(BaseModel):
    """Dataset paths and language configuration."""

    audio_dir: str
    train_csv: str
    eval_csv: str
    language: str


class TrainingConfig(BaseModel):
    """Training hyperparameters for fine-tuning."""

    batch_size: int
    eval_batch_size: int
    num_epochs: int
    learning_rate: float


class FinetuneConfig(BaseModel):
    """Top-level configuration for XTTS fine-tuning."""

    data: DataConfig
    training: TrainingConfig
    output_path: str
