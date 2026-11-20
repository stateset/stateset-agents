Environments
============

Environments define episode reset/step logic for training.

Typical entrypoint:

- `stateset_agents.core.environment.ConversationEnvironment`

Building environments by name
-----------------------------

``stateset_agents.core.environment_registry`` resolves an environment from a
name:

.. code-block:: python

    from stateset_agents.core.environment_registry import (
        available_environments,
        make_environment,
    )

    print(available_environments())
    environment = make_environment("refund-policy", scenarios=cases)

Publishing an environment from another package
----------------------------------------------

Declare an entry point in the ``stateset_agents.environments`` group whose
value is a factory returning an ``Environment``:

.. code-block:: toml

    [project.entry-points."stateset_agents.environments"]
    my-task = "my_package.envs:make_environment"

After ``pip install my-package`` the name ``my-task`` is available to
``make_environment`` with no change to this package.

Reward failures
---------------

By default a reward that cannot be computed is replaced by a fallback value,
counted in ``environment.reward_failures`` and flagged on the trajectory as
``metadata["reward_error"]``. Pass ``strict_rewards=True`` to the environment,
or set ``TrainingConfig(strict_rewards=True)``, to raise instead. Use strict
handling whenever a judge or external checker produces the reward: a silent
fallback is indistinguishable from a real score.
