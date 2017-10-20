========
smp\_sys
========

    :Author: Oswald Berthold



1 smp\_sys
----------

Part of the sensorimotor primitives (smp) ecosystem and designed to
work together with `smp\_base <https://github.com/x75/smp_base>`_ and `smp\_graphs <https://github.com/x75/smp_graphs>`_. The *sys* bit stands for
*systems*. In the smp context, a system is anything that produces data
and usually that is a robot, a different robot, or a file reader but
could also be a network feed (network monitoring, market data, social
network channels, ...), a wireless sensor network, a smart home, and
so on.

1.1 Implemented systems
~~~~~~~~~~~~~~~~~~~~~~~

- n-dimensional point mass (the simplest robot with motility,
  i.e. that can move it's base)

- simplearm (flowersteam's `explauto <https://github.com/flowersteam/explauto>`_)

- bha (bionic handling assistant model, python port by
  `Mathias Schmerling <https://github.com/gitmatti>`_ of the `bha matlab model <https://code.cor-lab.de/projects/goal-babbling-matlab>`_)

- more coming, stay tuned

1.2 Notes
~~~~~~~~~

- For BHA see also `https://www.cor-lab.de/bionic-handling-assistant <https://www.cor-lab.de/bionic-handling-assistant>`_
