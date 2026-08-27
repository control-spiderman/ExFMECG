"""Class registry used by the original ExFMECG training framework."""


class Registry:
    def __init__(self):
        self.builders = {}
        self.tasks = {}
        self.models = {}
        self.schedulers = {}
        self.runners = {}
        self.paths = {}
        self.state = {}

    @staticmethod
    def _decorator(mapping, name):
        def register(item):
            if name in mapping:
                raise KeyError(f"'{name}' is already registered")
            mapping[name] = item
            return item

        return register

    def register_builder(self, name):
        return self._decorator(self.builders, name)

    def register_task(self, name):
        return self._decorator(self.tasks, name)

    def register_model(self, name):
        return self._decorator(self.models, name)

    def register_lr_scheduler(self, name):
        return self._decorator(self.schedulers, name)

    def register_runner(self, name):
        return self._decorator(self.runners, name)

    def register_path(self, name, path):
        self.paths[name] = path

    def register(self, name, value):
        self.state[name] = value

    def get_builder_class(self, name):
        return self.builders.get(name)

    def get_task_class(self, name):
        return self.tasks.get(name)

    def get_model_class(self, name):
        return self.models.get(name)

    def get_lr_scheduler_class(self, name):
        return self.schedulers.get(name)

    def get_runner_class(self, name):
        return self.runners.get(name)

    def get_path(self, name):
        return self.paths.get(name)

    def get(self, name, default=None):
        return self.state.get(name, default)


registry = Registry()
