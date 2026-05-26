.. _finetune-examples:

Step-by-step reproduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| This example gallery contains tutorials for fine-tuning descending drive parameters to match
| target firing rates or force levels. Examples 01–04 form a sequential chain and must be run
| in numbered order, since each example loads results saved by its predecessor.
|
| **01** optimises the descending drive to reproduce a target motor-unit firing rate.
| **02** uses those optimised parameters to compute the resulting muscle force.
| **03** re-optimises the drive to hit a specific force target (e.g. 30 % MVC).
| **04** extracts inter-spike intervals and their coefficient of variation across contraction ramps.
