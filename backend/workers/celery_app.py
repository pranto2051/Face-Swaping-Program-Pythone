from celery import Celery

app = Celery('deepfake_studio')

app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/1',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,           # Re-deliver on worker crash
    worker_prefetch_multiplier=1,  # One job at a time per GPU worker
    task_reject_on_worker_lost=True,
    task_routes={
        'tasks.fast_swap': {'queue': 'gpu_fast'},
        'tasks.studio_swap': {'queue': 'gpu_heavy'},
        'tasks.cinematic_swap': {'queue': 'gpu_heavy'},
        'tasks.video_swap': {'queue': 'gpu_heavy'},
    }
)
