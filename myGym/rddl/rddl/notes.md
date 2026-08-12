# Notes for task generation

A little more in-depth since it apparently should be more general (handling task generation, not just rewards).

## Ideas

### Main idea

Elementary action (reward) vs Complex (Compound?) action (reward).

Elementary actions have methods `calculate` to get the current reward value. Additionally, they have initial and goal predicates (akin to pre-conditions and post-conditions).

### Radical views on actions, tasks & rewards

#### Separability of reward system

Action = reward + some extra functionality (checking execution). Should rewards be handled separately?

#### What is a task?

Task could be very complex (e.g., preparing a meal). Action is still very simple, even when compounded from several elementary actions.

Simple distinction: task is always a strict sequence of operations. Could be parallel but has very clearly separated operations (actions). That is, when an operation is finished, it is assumed to be finished "forever". 

#### Conclusions and definitions from the above

#### Taxonomy of activities

* atomic action (atom or proto-action)
  * basic action that cannot be effectively decomposed to simpler operations
  * typically represents basic skill of the execution system (basic robot functionality)

* composite action (complex action)
  * a combination of atomic actions (can actually be a mix of atomic and composite actions) but only up to a certain complexity level
  * the combination is not necessarily sequential
  * typically, only a single object is being handled
  * an important aspect is that the components are interlinked, i.e., the composition operators (that combine the actions) are checked "together" and if any of them fail, the action fails altogether
  * actions consider only general (classes of) objects, i.e., action definition restricts only a class (set of properties) of object, not specific object (category)
    * example: action `pick_n_place` might require the target to be pickable; however there can never be a requirement to pick for example "a fruit"

* task
  * activity composed of actions (atomic or composite)
  * typically, longer in duration and may involve several different objects
  * ordering of the components (actions) might not be strict (e.g., "put cereal into a bowl, then milk" and "put milk into bowl, then cereal" yields practically the same result)
  * components are logically separate, i.e., once a component (action) is done, there is no need to check its state (unlike with composite actions)
  * consequence of the previous is that if a component fails, it can potentially be redone without the need to redo the entire task (although, of course, some previous components might need to be redone in more complex cases, e.g., cooking)
  * a task might specify a specific object or object category (e.g., an apple or a fruit) to be targeted by the components

##### Conclusions from the above

* a composite action can be composited further into (more complex) composite action by combination with atomic or other composite actions
* all actions, atomic and composite, are also tasks but a task is not necessarily an action
* there is only a fuzzy border between composite actions and tasks
* the requirements for action to specify only a class of target objects (i.e., object properties) is only for their definitions. That is, of course, an instance of an action can target a specific object; for clarity:
  * `cut(X, Y)` can be an action of cutting X with Y
  * `X` must have the property `cuttable`
  * `Y` must have the property `has_sharp_edge`
  * `cut_an_apple(A, K)` can be a task that would require `K` to be some knife and `A` to be some apple
  * the task would be defined as:

  ```haskell
  cut_an_apple(A: Apple, K: Knife): cut(A, K)
  ```

  * thus, the task uses the (general) `cut` action to perform a more specific task of cutting of an apple
  * for convenience, action can specify entities by class name, instead of properties
    * common case: Gripper, Object, Location
      * these are, however, actually just shorthands for entities with certain properties, e.g., Gripper: entities with `can_grip` property
  * distinction between object category and class (property requirement) might be fuzzy
    * in general, swapping of categories does not necessarily result in failure of the task, as long, as the new category conforms to the class of the underlying action(s)
    * for example:
      * `cut_an_apple(A, K)` might be performed even if `A` is a banana (or any other `cuttable` object); although, the outcome might be slightly different
      * `cut(X, Y)` cannot will fail if `X` is not `cuttable`, e.g., if `X` is an iron ball, the cut action will have no effect and its post condition will never succeed

## Issues

### reach - reach (in Poke)

How to interpret? Reach is completed when gripper is near an object. Consequent reach should therefore just "exit" immediately without any action(?)
Maybe a "short push" or short movement should be used instead?

### Push - can it use 'move' element?

Push does not actually work with move, unless move is tailored to pushing.  
(note: reward is easy to define but to generate and evaluate a task is difficult)  
Problem:

```python
move(o, t):  # basic version
  move_gripper_towards(g, t)
```

But what if o is not in the gripper? And does not move with the gripper?

```python
move(o, t):  # ensuring object is held
  if holding(g, o):
    move_gripper_towards(g, t)
```

But what to do when not holding? Try to grasp? Fail?

```python
move(o, t):  # "intelligent" move
  if holding(g, o):
    move_gripper_towards(g, t)
  else:
    while not near(g, o):
      move_gripper_towards(g, o)
    grasp(g, o)
```

But this is kind of cheating - move is no longer elementary action!

```python
move(o, t):  # "assertive" move
  if holding(g, o):
    move_gripper_towards(g, t)
  else:
    fail()
```

This would be correct, but the problem is that this assumes objects can only be moved if they are grasped. And it is assumed objects can be moved if they are grasped (counter-example: large rock).

Possible implementation:

```python
move(o, t):
  move_gripper_towards(g, t)
  if somehow_handle_not_moving_object(o):
    fail()
```

Now both "in-hand" and "out-of-hand" moving works. But how to implement the `somehow_handle_not_moving_object` function? Instant-fail would be incorrect as it might take some time to accelerate heavy objects. Also, temporary stopping to move object (e.g., when pushing) can happen. Therefore, the function would have to handle "delayed movement with hysteresis". This will require several parameters and potentially require keeping history of object states.  
Is that a suitable approach?

## General questions

### How to compose an activity?

* difference between a reward and execution
  * ideally, the composition system would output an object that can be used to both execute the action/task and provide reward
  * but how?
