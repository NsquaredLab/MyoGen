.. _examples-index:

==================
Examples
==================

| MyoGen has a variety of examples demonstrating key functionalities, workflows, and research applications. 
| Each section contains detailed python examples that guide you through the implementation and usage of MyoGen components.
|
.. grid:: 1
    :gutter: 3

    .. grid-item::
        :columns: 12 6 6 4
        :margin: auto

        .. card::
            :link: basic-examples
            :link-type: ref
            :text-align: center

            Basics
            ^^^

            Self-contained examples covering core MyoGen components and workflows.

            +++
            For first-time users

.. grid:: 2
    :gutter: 3

    .. grid-item-card::
        :link: finetune-examples
        :link-type: ref
        :text-align: center

        Finetuning
        ^^^

        Workflow for matching MU and cortical population activity to match MVC % targets.

        +++
        Requires Optuna

    .. grid-item-card::
        :link: watanabe-reproduction
        :link-type: ref
        :text-align: center

        Literature Reproductions
        ^^^

        Reimplementations of published neuromuscular modeling studies.

        +++
        Computationally intensive

----

.. toctree::
   :hidden:
   :maxdepth: 2

   auto_examples/basic/index
   auto_examples/finetune/index
   auto_examples/papers/watanabe/index
