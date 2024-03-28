import os
import sys
from diffusers import pipelines

# Register models with diffusers, ensuring `from_pretrained` works
for filename in os.listdir(os.path.join(os.path.dirname(__file__), 'models')):
    if filename.endswith('.py') and not filename.startswith('__'):
        model_name = filename[:-3]
        sys.modules[f'{__name__}.models.{model_name}'] = __import__(f'{__name__}.models.{model_name}', fromlist=[model_name])
        setattr(pipelines, f'champ.models.{model_name}', sys.modules[f'{__name__}.models.{model_name}'])

# Re-export pipeline
from aniportrait.pipelines import AniPortraitPipeline
__all__ = ['AniPortraitPipeline']
