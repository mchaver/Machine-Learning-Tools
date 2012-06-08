package hmm;

//import java.util.ArrayList;
import java.util.ArrayList;

public class HMM {
	//constructor
	private double[][] B; //emission probabilites
	private double[][] A; //transition probabilites
	private double[] pi; //initial state probabilites
	
	private int symbols; //size of vocabulary
	private int states; //number of states

	private double[] coefficients;
	private double decodingProbability;
	
	public HMM(int symbols, int states) { //initiate without probability or emission
		this.symbols = symbols;
		this.states = states;
		//transitions
		A = new double[states][states];
		for (int i = 0; i < states; i++)
			for (int j = 0; j < states; j++)
				A[i][j] = 1.0 / states;
		//for (int i = 0; i < states; i++)
		//	for (int j = i; j < states; j++)
		//		A[i][j] = 1.0 / (states - i);
		
		//initial probability
		pi = new double[states];
		//emissions
		pi[0] = 1.0;
		//for (int i = 0; i < states; i++)
			//pi[i] = 1.0;
		
		B = new double[states][symbols];
		for (int i = 0; i < states; i++)
			for (int j = 0; j < symbols; j++)
				B[i][j] = 1.0 / symbols;
	}
	public HMM(int symbols, int states, double[] initialProb) {
		this.symbols = symbols;
		this.states = states;
		this.B = new double[states][symbols];
		this.pi = initialProb;
		for (int i = 0; i < states; i++)
			for (int j = 0; j < symbols; j++)
				B[i][j] = 1.0 / symbols;
	}
	//public HMM(double[][] emission, double[][] transition, double[][] initial ){
	//	B = emission;
	//	A = transition;
	//}
	
	public double evaluate(int[] observations, Boolean logarithm){
		if (observations.length == 0)
			return 0.0;
		
		double likelihood = 0;
		//double[] coefficients;// = new double[observations.length];
		
		forward(observations);
		
		for(int i = 0; i < coefficients.length;i++)
			likelihood += Math.log(coefficients[i]);
		
		return logarithm ? likelihood : Math.exp(likelihood);
	}
	
	private double[][] forward(int[] observations) {
		int T = observations.length;
		
		double[][] fwd = new double[T][states];
		coefficients = new double[T];
		
		//Initialization
		for (int i = 0; i < states; i++) {
			fwd[0][i] = pi[i] * B[i][observations[0]];
			coefficients[0] += fwd[0][i];
			//coefficients[0] += fwd[0][i] = pi[i] * B[i][observations[0]];
		}
		if(coefficients[0] != 0) {
			for (int i = 0; i < states; i++)
				fwd[0][i] = fwd[0][i] / coefficients[0];
		}
		
		//Induction
		for (int t = 1; t < T; t++) {
			for (int i = 0; i < states; i++) {
				double p = B[i][observations[t]];
				double sum = 0.0;
				for (int j = 0; j < states; j++)
					sum += fwd[t-1][j] * A[j][i];
				fwd[t][i] = sum * p;
				coefficients[t] += fwd[t][i];
				
			}
			
			if (coefficients[t] != 0) {
				for (int i = 0; i < states;i++)
					fwd[t][i] = fwd[t][i] /coefficients[t];
			}
		}
		
		return fwd;
	}
	
	private double[][] backward(int[] observations) {
		int T = observations.length;
		
		double[][] bwd = new double[T][states];
		
		for(int i = 0; i < states; i++)
			bwd[T-1][i] = 1.0 / coefficients[T-1];
		
		for (int t = T - 2; t >= 0; t--) {
			for (int i = 0; i < states; i++) {
				double sum = 0;
				for (int j = 0; j < states; j++) 
					sum += A[i][j] * B[j][observations[t+1]] * bwd[t+1][j];
				bwd[t][i] += sum / coefficients[t];
			}
		}
		
		return bwd;
	}
	
	private static boolean checkConvergence(double oldLikelihood, double newLikelihood, int currentIteration, int maxIterations, double tolerance) {
		if (tolerance > 0)
		{
			if (Math.abs(oldLikelihood - newLikelihood) <= tolerance)
				return true;
			
			if (maxIterations > 0) {
				if (currentIteration >= maxIterations)
					return true;
			}
		} else {
			if (currentIteration == maxIterations)
				return true;
		}
		
		if (Double.isNaN(newLikelihood) || Double.isInfinite(newLikelihood))
			return true;
		
		return false;
	}
	
	public int[] decode(int[] observations) {
		int T = observations.length;
		int minState;
		double minWeight;
		double weight;
		
		int[][] s = new int[states][T];
		double[][] a = new double[states][T];
		
		//Base
		for(int i = 0; i < states; i++) {
			a[i][0] = (-1.0 * Math.log(pi[i]) - Math.log(B[i][observations[0]]));
		}
		
		//Induction
		for (int t = 1; t < T; t++) {
			for (int j = 0; j < states; j++) {
				minState = 0;
				minWeight = a[0][t-1] - Math.log(A[0][j]);
				
				for (int i = 1; i < states; i++){
					weight = a[i][t-1] - Math.log(A[i][j]);
					if (weight < minWeight) {
						minState = i;
						minWeight = weight;
					}
				}
				
				a[j][t] = minWeight - Math.log(B[j][observations[t]]);
				s[j][t] = minState;
			}
		}
		
		minState = 0;
		minWeight = a[0][T-1];
		
		//Find the mininum value for time T-1
		for (int i = 1; i < states; i++) {
			if (a[i][T-1] < minWeight) {
				minState = i;
				minWeight = a[i][T-1];
			}
		}
		
		int[] path = new int[T];
		path[T-1] = minState;
		
		for (int t = T-2; t >= 0; t--)
			path[t] = s[path[t+1]][t+1];
		
		decodingProbability = Math.exp(-minWeight);
		return path;
	}
	
	public double learn(int[][] observations, int iterations, double tolerance) {
		if (iterations == 0 && tolerance == 0)
			return 0.0;
		
		int N = observations.length;
		int currentIteration = 1;
		boolean stop = false;
		
		//double[] pi = Probalities;
		//double[][] A = Transitions;
		
		double[][][][] epsilon = new double[N][][][];
		double[][][] gamma = new double[N][][];
		
		for (int i = 0; i < N;i++) {
			int T = observations[i].length;
			epsilon[i] = new double[T][states][states];
			gamma[i] = new double[T][states];
		}
		
		double oldLikelihood = Double.MIN_VALUE;
		double newLikelihood = 0;
		
		do{
			for(int i = 0; i < N; i++) {
				int[] sequence = observations[i];
				int T = sequence.length;

				//forward and backward probability for each state
				double[][] fwd = forward(observations[i]); //forward observations
				double[][] bwd = backward(observations[i]);

				//gamma values for next computation
				for (int t = 0; t < T; t++) {
					double s = 0;
					for (int k = 0; k < states; k++) {
						s += gamma[i][t][k] = fwd[t][k] * bwd[t][k];
					}
					if (s != 0){
						for (int k = 0; k < states; k++)
							gamma[i][t][k] /= s;
					}
				}
				
				//epsilon values
				for (int t = 0; t < T - 1; t++) {
					double s = 0;
					for(int k = 0; k < states; k++)
						for(int l = 0; l < states; l++)
							s += epsilon[i][t][k][l] = fwd[t][k] * A[k][l] * bwd[t+1][l] * B[l][sequence[t+1]];
					if(s != 0) {
						for (int k = 0; k < states; k++)
							for(int l = 0; l < states; l++)
								epsilon[i][t][k][l] /= s;
					}
				}
				
				//log-likelhood for sequence
				for (int t = 0; t < coefficients.length; t++)
					newLikelihood += Math.log(coefficients[t]);
			}
			
			//average likelihood for all sequences
			newLikelihood /= observations.length;
			
			if(checkConvergence(oldLikelihood, newLikelihood,currentIteration, iterations, tolerance)) {
				stop = true;
			} else {
				currentIteration++;
				oldLikelihood = newLikelihood;
				newLikelihood = 0.0;
				
				//re-estimate initial state probabilities
				for (int k = 0; k < states; k++) {
					double sum = 0;
					for (int i = 0; i < N; i++)
						sum += gamma[i][0][k];
					pi[k] = sum / N;
				}
				
				//re-estimate transition probabilities
				for (int i = 0; i < states; i++) {
					for (int j = 0; j < states; j++) {
						double den = 0;
						double num = 0;
						for(int k = 0; k < N; k++) {
							int T = observations[k].length;
							
							for (int l = 0; l < T - 1; l++)
								num += epsilon[k][l][i][j];
							
							for (int l = 0; l < T - 1; l++)
								den += gamma[k][l][i];
						}
						
						A[i][j] = (den != 0) ? num / den : 0.0;
					}
				}
				
				//re-estimate emission probabilities
				for (int i = 0; i < states; i++) {
					for (int j = 0; j < symbols; j++) {
						double den = 0;
						double num = 0;
						
						for (int k = 0; k < N; k++) {
							int T = observations[k].length;
							
							for (int l = 0; l < T; l++) {
								if (observations[k][l] == j)
									num += gamma[k][l][i];
							}
							
							for (int l = 0; l < T; l++)
								den += gamma[k][l][i];
						}
						
						B[i][j] = (num == 0) ? 1e-10 : num / den;
					}
				}
			}
			
		} while(!stop);
		
		return newLikelihood;
	}

	public static void main(String[] args){
		int[][] sequences = {{0,1,1,1,1,1,1},{0,1,1,1},{0,1,1,1,1},{0,1},{0,1,1}};
		HMM hmm = new HMM(2,2);
		hmm.learn(sequences, 0, 0.01);
		int[] trial1 = {0,1};
		int[] trial2 = {0,1,1,1};
		int[] trial3 = {1,1};
		int[] trial4 = {1,0,0,0};
		double test1[] = new double[4];
		test1[0] = hmm.evaluate(trial1 , false);
		test1[1] = hmm.evaluate(trial2 , false);
		test1[2] = hmm.evaluate(trial3 , false);
		test1[3] = hmm.evaluate(trial4 , false);
		for(int i = 0; i < 4; i++)
			System.out.println(test1[i]);
		
		int[][] sequences2 = {{ 0,1,1,1,1,0,1,1,1,1 }, { 0,1,1,1,0,1,1,1,1,1 },{ 0,1,1,1,1,1,1,1,1,1 },{ 0,1,1,1,1,1},{ 0,1,1,1,1,1,1},{ 0,1,1,1,1,1,1,1,1,1},{ 0,1,1,1,1,1,1,1,1,1 }};
		HMM hmm2 = new HMM(2,3);
		hmm2.learn(sequences2, 0, 0.0001);
		int[] trial5 = {0,1};
		int[] trial6 = {0,1,1,1};
		int[] trial7 = {1,1};
		int[] trial8 = {1,0,0,0};
		int[] trial9 = {0,1,0,1,1,1,1,1,1};
		int[] trial10 = {0,1,1,1,1,1,1,0,1};
		double test2[] = new double[6];
		
		test2[0] = hmm2.evaluate(trial5, false);
		test2[1] = hmm2.evaluate(trial6, false);
		test2[2] = hmm2.evaluate(trial7, false);
		test2[3] = hmm2.evaluate(trial8, false);
		test2[4] = hmm2.evaluate(trial9, false);
		test2[5] = hmm2.evaluate(trial10, false);
		for(int i = 0; i < 6; i++)
			System.out.println(test2[i]);
	}
}
