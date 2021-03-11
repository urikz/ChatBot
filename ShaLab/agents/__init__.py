import logging
import os

from ShaLab.engine import Engine
from ShaLab.data import (
    DialogCorpusWithProfileMemory,
    DialogCorpus,
    WordVocabulary,
)
from ShaLab.models import create_from_checkpoint

from .model_based_agent import ModelBasedAgent
from .random_agents import RandomWordsAgent, RandomSentencesAgent


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def _load_corpus_hacky(path):
    try:
        return DialogCorpusWithProfileMemory.from_file(path)
    except:
        try:
            return DialogCorpus.from_file(path)
        except:
            return None


def create_agent(path, policy, existing_models, args):
    if os.path.isdir(path):
        path = Engine.get_best_chechpoint(path)
    if path not in existing_models:
        model = create_from_checkpoint(path, args.gpu)
        model.eval()
        existing_models[path] = model

    return ModelBasedAgent(
        model=existing_models[path],
        max_length=args.max_length,
        policy=policy,
        num_candidates=1,
        context=None,
    )
