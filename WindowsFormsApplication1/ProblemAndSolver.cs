using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Diagnostics;
using System.Linq;


namespace TSP
{

	class ProblemAndSolver
	{

		private class TSPSolution
		{
			/// <summary>
			/// we use the representation [cityB,cityA,cityC] 
			/// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
			/// and the edge from cityC to cityB is the final edge in the path.  
			/// You are, of course, free to use a different representation if it would be more convenient or efficient 
			/// for your data structure(s) and search algorithm. 
			/// </summary>
			public ArrayList
				Route;

			/// <summary>
			/// constructor
			/// </summary>
			/// <param name="iroute">a (hopefully) valid tour</param>
			public TSPSolution(ArrayList iroute)
			{
				Route = new ArrayList(iroute);
			}

			/// <summary>
			/// Compute the cost of the current route.  
			/// Note: This does not check that the route is complete.
			/// It assumes that the route passes from the last city back to the first city. 
			/// </summary>
			/// <returns></returns>
			public double costOfRoute()
			{
				// go through each edge in the route and add up the cost. 
				int x;
				City here;
				double cost = 0D;

				for (x = 0; x < Route.Count - 1; x++)
				{
					here = Route[x] as City;
					cost += here.costToGetTo(Route[x + 1] as City);
				}

				// go from the last city to the first. 
				here = Route[Route.Count - 1] as City;
				cost += here.costToGetTo(Route[0] as City);
				return cost;
			}
		}

		#region Private members 

		/// <summary>
		/// Default number of cities (unused -- to set defaults, change the values in the GUI form)
		/// </summary>
		// (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
		// click on the Problem Size text box, go to the Properties window (lower right corner), 
		// and change the "Text" value.)
		private const int DEFAULT_SIZE = 25;

		/// <summary>
		/// Default time limit (unused -- to set defaults, change the values in the GUI form)
		/// </summary>
		// (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
		// click on the Time text box, go to the Properties window (lower right corner), 
		// and change the "Text" value.)
		private const int TIME_LIMIT = 60;		//in seconds

		private const int CITY_ICON_SIZE = 5;


		// For normal and hard modes:
		// hard mode only
		private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

		/// <summary>
		/// the cities in the current problem.
		/// </summary>
		private City[] Cities;
		/// <summary>
		/// a route through the current problem, useful as a temporary variable. 
		/// </summary>
		private ArrayList Route;
		/// <summary>
		/// best solution so far. 
		/// </summary>
		private TSPSolution bssf; 

		/// <summary>
		/// how to color various things. 
		/// </summary>
		private Brush cityBrushStartStyle;
		private Brush cityBrushStyle;
		private Pen routePenStyle;


		/// <summary>
		/// keep track of the seed value so that the same sequence of problems can be 
		/// regenerated next time the generator is run. 
		/// </summary>
		private int _seed;
		/// <summary>
		/// number of cities to include in a problem. 
		/// </summary>
		private int _size;

		/// <summary>
		/// Difficulty level
		/// </summary>
		private HardMode.Modes _mode;

		/// <summary>
		/// random number generator. 
		/// </summary>
		private Random rnd;

		/// <summary>
		/// time limit in milliseconds for state space search
		/// can be used by any solver method to truncate the search and return the BSSF
		/// </summary>
		private int time_limit;
		#endregion

		#region Public members

		/// <summary>
		/// These three constants are used for convenience/clarity in populating and accessing the results array that is passed back to the calling Form
		/// </summary>
		public const int COST = 0;		   
		public const int TIME = 1;
		public const int COUNT = 2;
		
		public int Size
		{
			get { return _size; }
		}

		public int Seed
		{
			get { return _seed; }
		}
		#endregion

		#region Constructors
		public ProblemAndSolver()
		{
			this._seed = 1; 
			rnd = new Random(1);
			this._size = DEFAULT_SIZE;
			this.time_limit = TIME_LIMIT * 1000;				  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

			this.resetData();
		}

		public ProblemAndSolver(int seed)
		{
			this._seed = seed;
			rnd = new Random(seed);
			this._size = DEFAULT_SIZE;
			this.time_limit = TIME_LIMIT * 1000;				  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

			this.resetData();
		}

		public ProblemAndSolver(int seed, int size)
		{
			this._seed = seed;
			this._size = size;
			rnd = new Random(seed);
			this.time_limit = TIME_LIMIT * 1000;						// TIME_LIMIT is in seconds, but timer wants it in milliseconds

			this.resetData();
		}
		public ProblemAndSolver(int seed, int size, int time)
		{
			this._seed = seed;
			this._size = size;
			rnd = new Random(seed);
			this.time_limit = time*1000;						// time is entered in the GUI in seconds, but timer wants it in milliseconds

			this.resetData();
		}
		#endregion

		#region Private Methods

		/// <summary>
		/// Reset the problem instance.
		/// </summary>
		private void resetData()
		{

			Cities = new City[_size];
			Route = new ArrayList(_size);
			bssf = null;

			if (_mode == HardMode.Modes.Easy)
			{
				for (int i = 0; i < _size; i++)
					Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
			}
			else // Medium and hard
			{
				for (int i = 0; i < _size; i++)
					Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
			}

			HardMode mm = new HardMode(this._mode, this.rnd, Cities);
			if (_mode == HardMode.Modes.Hard)
			{
				int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
				mm.removePaths(edgesToRemove);
			}
			City.setModeManager(mm);

			cityBrushStyle = new SolidBrush(Color.Black);
			cityBrushStartStyle = new SolidBrush(Color.Red);
			routePenStyle = new Pen(Color.Blue,1);
			routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
		}

		#endregion

		#region Public Methods

		/// <summary>
		/// make a new problem with the given size.
		/// </summary>
		/// <param name="size">number of cities</param>
		public void GenerateProblem(int size, HardMode.Modes mode)
		{
			this._size = size;
			this._mode = mode;
			resetData();
		}

		/// <summary>
		/// make a new problem with the given size, now including timelimit paremeter that was added to form.
		/// </summary>
		/// <param name="size">number of cities</param>
		public void GenerateProblem(int size, HardMode.Modes mode, int timelimit)
		{
			this._size = size;
			this._mode = mode;
			this.time_limit = timelimit*1000;  //convert seconds to milliseconds
			resetData();
		}

		/// <summary>
		/// return a copy of the cities in this problem. 
		/// </summary>
		/// <returns>array of cities</returns>
		public City[] GetCities()
		{
			City[] retCities = new City[Cities.Length];
			Array.Copy(Cities, retCities, Cities.Length);
			return retCities;
		}

		/// <summary>
		/// draw the cities in the problem.  if the bssf member is defined, then
		/// draw that too. 
		/// </summary>
		/// <param name="g">where to draw the stuff</param>
		public void Draw(Graphics g)
		{
			float width  = g.VisibleClipBounds.Width-45F;
			float height = g.VisibleClipBounds.Height-45F;
			Font labelFont = new Font("Arial", 10);

			// Draw lines
			if (bssf != null)
			{
				// make a list of points. 
				Point[] ps = new Point[bssf.Route.Count];
				int index = 0;
				foreach (City c in bssf.Route)
				{
					if (index < bssf.Route.Count -1)
						g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
					else 
						g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
					ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
				}

				if (ps.Length > 0)
				{
					g.DrawLines(routePenStyle, ps);
					g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
				}

				// draw the last line. 
				g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
			}

			// Draw city dots
			foreach (City c in Cities)
			{
				g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
			}

		}

		/// <summary>
		///  return the cost of the best solution so far. 
		/// </summary>
		/// <returns></returns>
		public double costOfBssf ()
		{
			if (bssf != null)
				return (bssf.costOfRoute());
			else
				return -1D; 
		}

		/// <summary>
		/// This is the entry point for the default solver
		/// which just finds a valid random tour 
		/// </summary>
		/// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
		public string[] defaultSolveProblem()
		{
			int i, swap, temp, count=0;
			string[] results = new string[3];
			int[] perm = new int[Cities.Length];
			Route = new ArrayList();
			Random rnd = new Random();
			Stopwatch timer = new Stopwatch();

			timer.Start();

			do
			{
				for (i = 0; i < perm.Length; i++) // create a random permutation template
					perm[i] = i;
				for (i = 0; i < perm.Length; i++)
				{
					swap = i;
					while (swap == i)
						swap = rnd.Next(0, Cities.Length);
					temp = perm[i];
					perm[i] = perm[swap];
					perm[swap] = temp;
				}
				Route.Clear();
				for (i = 0; i < Cities.Length; i++)	// Now build the route using the random permutation 
				{
					Route.Add(Cities[perm[i]]);
				}
				bssf = new TSPSolution(Route);
				count++;
			} while (costOfBssf() == double.PositiveInfinity);				// until a valid route is found
			timer.Stop();

			results[COST] = costOfBssf().ToString();						  // load results array
			results[TIME] = timer.Elapsed.ToString();
			results[COUNT] = count.ToString();

			return results;
		}

		/// <summary>
		/// performs a Branch and Bound search of the state space of partial tours
		/// stops when time limit expires and uses BSSF as solution
		/// </summary>
		/// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
		public string[] bBSolveProblem()
		{
			string[] results = new string[3];
			Stopwatch timer = new Stopwatch();
			timer.Start();

			double[,] costMatrix = getCityMatrix(Size);
			State myBSSF = new State(new ArrayList(), costMatrix, Double.MaxValue);
			HeapQueue pQueue = new HeapQueue(1000000);

			//Set up matrix and initial state
			ArrayList rt = new ArrayList();
			rt.Add(0); //start at city 0.
			double[,] startMatrix = (double[,])costMatrix.Clone();
			double startBound = reduceMatrix(startMatrix, Size);
			State start = new State(rt, startMatrix, startBound);
			pQueue.insert(start);

			//set BSSF to bound find by greedy algorithm to improve pruning
			myBSSF.bound = doGreedy(start);

			//keep track of solutions, pruned states, states created, etc. 
			int excluded = 0;
			int solutionCounter = 0;
			int stateCounter = 0;

			//Main branch and bound loop. Theoretically the worst-case for this is O(n!) because it's possible the algorithm can do very little pruning.
			while (!pQueue.isEmpty() && timer.ElapsedMilliseconds < time_limit) {
				State currentState = pQueue.deleteMin(); //Choose the most promising state to try next.			
				int currCityIdx = (int)currentState.Route[currentState.Route.Count - 1];
				for (int nextCityIdx = 0; nextCityIdx < Size; nextCityIdx++) {
					//select unvisited city to go to.
					if (!(currentState.Route.Contains(nextCityIdx))) {
						ArrayList route = (ArrayList)currentState.Route.Clone();
						double[,] newMatrix = (double[,])currentState.matrix.Clone();
						double currentBound = currentState.bound;

						//cost to take this edge
						currentBound += newMatrix[currCityIdx, nextCityIdx];

						//'remove' row(currCityIdx) and column(i) by setting their cost to the maximum value. 
						for (int j = 0; j < Size; j++) {
							newMatrix[currCityIdx, j] = Double.MaxValue;
							newMatrix[j, nextCityIdx] = Double.MaxValue;
						}
						newMatrix[nextCityIdx, currCityIdx] = Double.MaxValue;
						route.Add(nextCityIdx);

						//reduce matrix and find the bound increase
						double increase = reduceMatrix(newMatrix, Size);

						State childState = new State(route, newMatrix, (currentBound + increase));
						stateCounter++;

						if (childState.Route.Count == Size) {
							if (childState.bound < myBSSF.bound) {
								myBSSF = childState; //update BSSF
								solutionCounter++;
								pQueue.prune(myBSSF.bound); //Prune states in queue with higher bounds than new BSSF.
							}
						}
						else if (myBSSF.bound >= childState.bound) {
							pQueue.insert(childState); //Add state to queue
						}
						else {
							excluded++;
						}
					}
				}
			}

			//I accidentally did my Routes storing integers rather than cities, so these next few lines just turn a State Route into a TSPSolution Route that can be drawn. 
			ArrayList rList = new ArrayList(Size);
			for (int i = 0; i < myBSSF.Route.Count; i++) {
				int cityIndex = (int)myBSSF.Route[i];
				rList.Insert(i, Cities[cityIndex]);
			}
			if (bssf == null) {
				bssf = new TSPSolution(new ArrayList());
			}
			bssf.Route = rList;

			timer.Stop();
			results[COST] = bssf.costOfRoute().ToString();  // load results into array here, replacing these dummy values
			results[TIME] = timer.Elapsed.ToString();
			results[COUNT] = solutionCounter.ToString();
			return results;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////
		// These additional solver methods will be implemented as part of the group project.
		////////////////////////////////////////////////////////////////////////////////////////////

		/// <summary>
		/// finds the greedy tour starting from each city and keeps the best (valid) one
		/// </summary>
		/// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
		public string[] greedySolveProblem()
		{
			string[] results = new string[3];

			int solutionCounter = 0;

			Stopwatch timer = new Stopwatch();

			timer.Start();

			// TODO: Add your implementation for a greedy solver here.
			Random rnd = new Random();
			List<City> remaining;
			for (int i = 0; i < this.GetCities().Length; i++) // Until we get a valid path or bust
			{
				Route = new ArrayList();
				remaining = new List<City>(this.GetCities());

				City s = remaining[i++]; // grab a start city
				remaining.Remove(s);
				Route.Add(s);
				while (remaining.Count != 0) // while there are cities not visited
				{
					double lowestCost = double.PositiveInfinity;
					City n = null;
					foreach (City c in remaining)
					{
						double cost = (Route[Route.Count - 1] as City).costToGetTo(c);
						if (cost < lowestCost)
						{
							lowestCost = cost;
							n = c;
						}
					}

					if (n == null) { break; } // If no route could be found, break out and try another node

					remaining.Remove(n);
					Route.Add(n);

				}

				if (remaining.Count != 0) { continue; } // If this route could not be completed

				solutionCounter++;

				TSPSolution sol = new TSPSolution(Route);
				if (this.bssf == null || sol.costOfRoute() < this.bssf.costOfRoute())
				{
					this.bssf = new TSPSolution(Route); // If a complete, better route was generated, plug it in.
				}
			}
			timer.Stop();

			results[COST] = costOfBssf().ToString();	// load results array
			results[TIME] = timer.Elapsed.ToString();
			results[COUNT] = solutionCounter.ToString();

			return results;
		}

		/**
		 * City sorting comparator
		 * O(1)
		 */ 
		private static int sortCities(City a, City b)
		{
			if (a == null && b == null) { return 0; }
			else if (a == null || a.X < b.X) { return -1; }
			else if (b == null || a.X > b.X) { return 1; }
			return 0;
		}
		/**
		 * Main structure used for creating the path
		 * For complexity I'll assume the "List" class is based on a linked-list. It's probably not.
		 * remove: O(c)
		 * add: O(c)
		 * distance: O(n)
		 */
		private class PathBuilder
		{
			// Fields
			private List<City> route= new List<City>();
			// Constructors
			public PathBuilder()
			{
			}
			public PathBuilder(int size)
			{
				//route.Capacity = size;
			}
			// Exports to correct format for bssf constructor
			public ArrayList export()
			{
				ArrayList output = new ArrayList();
				for (int i = 0; i < route.Count; i++) output.Add(route[i]);
				return output;
			}
			// Add object to list
			public void add(City to_add){
				route.Add(to_add);
			}
			public void add(City to_add, int index_after) {
				route.Insert(index_after + 1, to_add);
			}
			// Gives the minimum distance away a city is from the hull.
			// return Tuple<best cost, insert position>
			public Tuple<double, int> distance(City city) {
				// Iterate through each node and find best distance.
				double best_cost = double.PositiveInfinity;
				int best_position=-1;
				for (int iter = 0; iter < route.Count; iter++)
				{
					double test_cost = this[iter].costToGetTo(city)+city.costToGetTo(this[iter+1])-this[iter].costToGetTo(this[iter+1]);
					if (test_cost < best_cost)
					{
						best_position = iter; // Set best distance
						best_cost = test_cost;
					}
				}
				// Return distance
				return new Tuple<double, int>(best_cost, best_position);
			}
			// Create index-style getter/setter
			public City this[int index]{
				get
				{
					while (index >= route.Count)
					{ // Enforce periodic boundary conditions
						index -= route.Count;
					}
					return route[index];
				}
				set
				{
					while (index >= route.Count)
					{ // Enforce periodic boundary conditions
						index -= route.Count;
					}
					route[index] = value;
				}
			}
			// Print contents
			public void debug() {
				Console.WriteLine("****Route****");
				for (int iter = 0; iter < route.Count; iter++)
				{
					Console.WriteLine("City: Index {0}, Position ({1}, {2})",iter, this[iter].X, this[iter].Y);
					Console.WriteLine("Path: Cost {0}, Vertices ({1}, {2}) -> ({3}, {4})", this[iter].costToGetTo(this[iter + 1]), this[iter].X, this[iter].Y, this[iter+1].X, this[iter+1].Y);
				}
				Console.WriteLine("Size: {0}", this.size());
				Console.WriteLine("*************");
			}
			// Getter for size
			public int size()
			{
				return route.Count;
			}
		}
		/**
		 * Improved Greedy Solution Strategy
		 * Total: O(n^2 log n)
		 * 
		 * Steps
		 * 1) Clone input: O(n)
		 * 2) Find initial triangle: O(n)
		 * 3) Add on all other nodes: O(n)*O(j)*O(k) = O(n)*O(j)*(n-j)~O(n)*O(n log n) = O(n^2 log n)
		 * 3.1) While there is free node: O(n)
		 * 3.1.1) Cycle through all free points: O(j) where n is number free.
		 * 3.1.1.1) Find distance from free point to hull: O(k) where n is number in hull.
		 * 4) Return result: O(c)
		 * 
		 */
		private ArrayList ImprovedGreedy(List<City> input, int first_index)
		{
			// Initialize variables
			List<City> cities = new List<City>(input);
			PathBuilder path_builder = new PathBuilder(cities.Count);
			// Initialize start city
			path_builder.add(cities[first_index]);
			cities.RemoveAt(first_index);
			// Find the closest points for the starting triangle
			Tuple<int, double> first_best = new Tuple<int, double>(-1, double.PositiveInfinity); // Tuple<index, cost>
			Tuple<int, double> second_best = new Tuple<int, double>(-1, double.PositiveInfinity);
			for (int iter = 0; iter < cities.Count; iter++) {
				double cost=path_builder[0].costToGetTo(cities[iter]);
				if (cost < second_best.Item2) { // Found a better item
					if (cost < first_best.Item2) { // Better item is better than both previous
						second_best = first_best;
						first_best= new Tuple<int, double>(iter, cost);
					}
					else
					{
						second_best = new Tuple<int, double>(iter, cost);
					}
				}
			}
			// Make sure we didn't start on an un-reachable node.
			if (first_best.Item2 != double.PositiveInfinity && second_best.Item2 != double.PositiveInfinity)
			{
				path_builder.add(cities[first_best.Item1]);
				path_builder.add(cities[second_best.Item1]);
			}
			else
			{ // Throw error if start node is unreachable.
				Console.WriteLine("Encountered Error in Algorithm");
				throw new Exception();
				// Unreachable node found
				// TODO: implement what to do here.
			}
			// Delete the last index first so we don't get a null exception.
			if (first_best.Item1 > second_best.Item1)
			{
				cities.RemoveAt(first_best.Item1);
				cities.RemoveAt(second_best.Item1);
			}
			// Now we can delete the first index.
			else
			{
				cities.RemoveAt(second_best.Item1);
				cities.RemoveAt(first_best.Item1);
			}
			// Add all cities to builder one at a time
			while (cities.Count > 0)
			{
				// Iterate through all unused cities and find the closest one.
				double best_cost = double.PositiveInfinity;
				int best_position = -1;
				int best_insert = -1;
				for (int iter = 0; iter < cities.Count; iter++)
				{
					Tuple<double, int> test=path_builder.distance(cities[iter]);
					if (test.Item1 < best_cost)
					{ // A closer node has been found.
						best_cost = test.Item1;
						best_position = iter;
						best_insert = test.Item2;
					}
				}
				// Add the closest node if there's a reachable one.
				if (best_position != -1)
				{
					path_builder.add(cities[best_position], best_insert);
					cities.RemoveAt(best_position);
				}
				else
				{
					Console.WriteLine("Encountered Error in Algorithm");
					throw new Exception();
					// Unreachable node found
					// TODO: implement what to do here.
				}
			}
			// Finialize problem
			return path_builder.export();
		}
		/**
		 * Greedy Divide and Conquer Algorithm
		 * O(???)
		 * 
		 * Notes:
		 * 
		 */ 
		public string[] fancySolveProblem()
		{
			string[] results = new string[3];
			Stopwatch timer = new Stopwatch();

			// TODO: Add your implementation for your advanced solver here.

			// Sort the array by X values. I'm assuming that it's O(n log(n) ) - Calvin
			//Array.Sort<City>(Cities, sortCities);

			// Algorithm Proper
			Random random = new Random();
			double best_cost=double.PositiveInfinity;
			int count=0;
			// Make a new data structre for the cities to be used.
			List<City> cities = new List<City>(Cities);
			// Start algorithm
			timer.Start();
			// Make a new structure to keep track of start nodes.
			List<int> start_indexes= new List<int>();
			for (int iter = 0; iter < cities.Count; iter++)
			{
				start_indexes.Add(iter);
			}

			// Iterate until we run out of time.
			while (timer.ElapsedMilliseconds < time_limit && start_indexes.Count > 0) {
				// Find next start index.
				int rnd_index = random.Next(start_indexes.Count);
				int first_index = start_indexes[rnd_index];
				start_indexes.RemoveAt(rnd_index);
				// Solve with random index.
				TSPSolution current = new TSPSolution(this.ImprovedGreedy(cities, first_index));
				count++;
				// Test to see if this is a better solution.
				double current_cost = current.costOfRoute();
				if (current_cost < best_cost) { // Found a better solution.
					bssf = current;
					best_cost = current_cost;
				}
			}
			// Ran out of time. Return results.
			timer.Stop();
			results[COST] = costOfBssf().ToString();
			results[TIME] = timer.Elapsed.ToString();
			results[COUNT] = count.ToString();
			return results;
		}


		//All of this stuff below is just from my branch and bound.--Jarett 

		//Makes the initial matrix with infinity (maxValue) along the diagonal and cost to get from city i to city j in cityMatrix[i, j]. O(n^2) time.
		public double[,] getCityMatrix(int size) {
			double[,] cityMatrix = new double[size, size];
			for (int i = 0; i < size; i++) {
				for (int j = 0; j < size; j++) {
					double dist = Cities[i].costToGetTo(Cities[j]);
					cityMatrix[i, j] = (i == j || Double.IsPositiveInfinity(dist)) ? Double.MaxValue : dist;
				}
			}
			return cityMatrix;
		}

		//Reduces the matrix so every row and column has a 0, and returns the increase that is added to the lowerbound for this reduction. Every cell of the matrix is visited so it has time complexity of O(n^2).
		public double reduceMatrix(double[,] matrix, int size) {
			double boundIncrease = reduceRowOrColumn(matrix, size, true); //reduce the rows
			boundIncrease += reduceRowOrColumn(matrix, size, false); //reduce the columns
			return boundIncrease;
		}
		//If doRow == true this reduces rows, otherwise reduces columns. Time complexity of O(n^2)
		public double reduceRowOrColumn(double[,] matrix, int size, bool doRow) {
			double boundIncrease = 0;
			for (int i = 0; i < size; i++) {
				double lowest = (doRow) ? matrix[i, 0] : matrix[0, i];
				//sweep through row/column looking for lowest value.
				for (int j = 0; j < size; j++) {
					if (doRow) {
						if (matrix[i, j] < lowest) {
							lowest = matrix[i, j];
						}
					}
					else {
						if (matrix[j, i] < lowest) {
							lowest = matrix[j, i];
						}
					}
				}
				//if lowest value is not 0 or maxvalue, reduce entire row/column by it. 
				if (lowest != 0 && lowest != Double.MaxValue) {
					boundIncrease += lowest;
					for (int j = 0; j < size; j++) {
						if (doRow) {
							if (matrix[i, j] != Double.MaxValue) {
								matrix[i, j] -= lowest;
							}
						}
						else {
							if (matrix[j, i] != Double.MaxValue) {
								matrix[j, i] -= lowest;
							}
						}
					}
				}
			}
			return boundIncrease;
		}

		/*Uses a greedy approach to find a short path by always taking the edge to the next closest, unvisited city. 
		 * This method finds the greedy route but only returns the bound which is used for pruning B&B. 
		 * Traverses entire matrix, so time complexity O(n^2).
		 */
		private double doGreedy(State state) {
			double bound = state.bound;
			ArrayList Route = (ArrayList)state.Route.Clone();
			double[,] matrix = (double[,])state.matrix.Clone();
			while (Route.Count < Size) {
				//get current city in route
				int currCityIdx = (int)Route[Route.Count - 1];
				double shortestPath = Double.MaxValue;
				int closestCityIndex = -1;
				//greedily choose next city in route
				for (int i = 0; i < Size; i++) {
					if (!(Route.Contains(i))) {
						double pathCost = matrix[currCityIdx, i];
						if (pathCost < shortestPath && !Double.IsPositiveInfinity(pathCost)) {
							shortestPath = pathCost;
							closestCityIndex = i;
						}
					}
				}
				//if all cities have been visited, add cost of path from this city to start city (0).
				if (Route.Count == Size - 1) {
					bound += matrix[currCityIdx, 0];
				}
				if (shortestPath != Double.MaxValue) {
					bound += shortestPath;
					Route.Add(closestCityIndex);
				}
			}
			return bound;
		}

		//Each node in the B&B tree is a state. Keeps track of bound, route, matrix, etc.  
		private class State {
			public ArrayList Route;
			public double bound;
			public double[,] matrix;
			private int capacity;
			private static double UNVISITED_WEIGHT = 250;
			public State(ArrayList iroute, double[,] m, double b) {
				Route = new ArrayList(iroute);
				bound = b;
				matrix = m;
				capacity = matrix.GetLength(0);
			}

			//Gives a weight to unvisited cities, which encourages the 'choosing' function of the heap to 
			// go deeper into the tree so that it's heuristically pruned. 
			public double weightedBound() {
				int unvisited = (this.capacity - Route.Count);
				return bound + (UNVISITED_WEIGHT * unvisited);
			}
		}

		//HeapQueue taken from project 3 but used to hold and sort State objects as well as 
		//information such as total of pruned states and max # of states. All methods are O(log n) except pruning which is O(n log n).
		private class HeapQueue {
			State[] heapArray;
			public int currentSize;
			bool empty;
			int capacity;
			public int totalPruned;
			public int maxSize;
			public HeapQueue(int count) {
				heapArray = new State[count + 1];
				currentSize = 0;
				totalPruned = 0;
				capacity = count;
				empty = true;
				maxSize = 0;
			}

			public void prune(double bound) {
				int removed = 0;
				for (int i = 1; i < currentSize; i++) {
					if (heapArray[i].bound >= bound) {
						deleteAt(i);
						removed++;
					}
				}
				totalPruned += removed;
			}

			public void insert(State node) {
				currentSize++;
				if (currentSize > maxSize) {
					maxSize = currentSize;
				}
				empty = false;
				bubbleUp(node, currentSize);
			}

			// Insert into highest index of array, then bubble it up to where it needs to be. O(log(n)).public void insert(State node) { currentSize++; empty = false; bubbleUp(node, currentSize);}/* O(log(n)). The node at heapArray[1] will always be the minimum, so replace it withthe last node in the heap, then sift that node down to its proper position.*/
			public State deleteMin() {
				State result = heapArray[1];

				heapArray[1] = heapArray[currentSize];
				currentSize--;
				empty = (currentSize == 0);
				if (!empty) {
					siftDown(heapArray[1], 1);
				}

				return result;
			}

			public void deleteAt(int i) {
				heapArray[i] = heapArray[currentSize];
				currentSize--;
				empty = (currentSize == 0);
				if (!empty) {
					siftDown(heapArray[i], i);
				}
			}

			public void bubbleUp(State node, int i) {
				int p = (i / 2);

				while (i != 1 && heapArray[p].weightedBound() > node.weightedBound()) {
					heapArray[i] = heapArray[p];
					i = p; p = (i / 2);
				}
				heapArray[i] = node;
			}
			public void siftDown(State node, int i) {
				int c = minChild(i);
				while (c != 1 && (heapArray[c].weightedBound() < node.weightedBound())) {
					heapArray[i] = heapArray[c];
					i = c; c = minChild(i);
				}
				heapArray[i] = node;
			}

			public int minChild(int i) {
				int leftPos = (2 * i);
				int rightPos = (2 * i) + 1;
				if (leftPos > currentSize) {
					return 1;
				}
				else if (leftPos == currentSize) {
					return (2 * i);
				}
				else {
					State leftChild = heapArray[leftPos];
					State rightChild = heapArray[rightPos];

					if (leftChild.bound == rightChild.bound) {
						return (leftChild.Route.Count > rightChild.Route.Count) ? leftPos : rightPos;
					}
					return (leftChild.weightedBound() < rightChild.weightedBound()) ? leftPos : rightPos;
				}
			}
			public bool isEmpty() { return empty; }
		}

		#endregion
	}

}
