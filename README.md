# w17-group-205

## Algorithm choice
Greedy Divide and Conquer

## The Gist of Things - Calvin's 2 cents
So here is what I see happening:
1. Sort all cities from left to right
2. Break the set of cities down into groups of 3-5 cities and solve with iterative greedy runs until we get the optimal.
3. From there, recombine our mini-circuits back into a while

In thinking about it, I feel that perhaps the best merge thought is to combine the leftmost vertex of the right circuit with the closest vertex to it in the left circuit. Then repeat that process with the second-to-leftmost vertex in the right circuit with the vertex in the right circuit (minus the one already picked) closest to it.
