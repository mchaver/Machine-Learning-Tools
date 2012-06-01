package perceptron;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Perceptron {
	private double threshold = 0.5;
	private double learningRate = 0.1;
	private double[] weights;
	
	private int[][] input;
	private int[] output;
	
	public int sizeOfInputVectors;
	public int numberOfInputVectors;
	
	public void setLearningRate(double learningRate){
		if(learningRate > 0 && learningRate <= 1){
			this.learningRate = learningRate;
		} else {
			throw new RuntimeException("Learning rate must be greater than 0 and less than or equal to 1.");
		}
	}
	public void setThreshold(double threshold){
		this.threshold = threshold;
	}
	
	private int sumFunction(int[] inputVector){
		int sum = 0;
		for(int i = 0; i < inputVector.length; i++) {
			sum += inputVector[i] * weights[i];
		}
		return sum;
	}

	private void readPerceptronFile(String filename) throws IOException{
		FileReader fileReader = new FileReader(filename);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		List<String> lines = new ArrayList<String>();
		String line = null;
		while ((line = bufferedReader.readLine()) != null) {
			lines.add(line);
		}
		bufferedReader.close();
		String[] entries = lines.toArray(new String[lines.size()]);
		String[] weightsString = entries[0].split(",");
		
		weights = new double[weightsString.length];
		for(int i = 0; i < weightsString.length; i++) {
			weights[i] = Double.parseDouble(weightsString[i]);
		}
		
		input = new int[entries.length-1][];
		output = new int[entries.length-1];
		for(int i = 0; i < input.length; i++) {
			input[i] = new int[weights.length];
		}
		
		for(int i = 1; i < entries.length;i++){
			System.out.println(entries[i]);
			String[] split = entries[i].split(";");
			output[i-1] = Integer.parseInt(split[1]);
			String[] inputString = split[0].split(",");
			for(int j = 0; j < inputString.length; j++){
				input[i-1][j] = Integer.parseInt(inputString[j]);
			}
		}
		
		sizeOfInputVectors = input[0].length;
		numberOfInputVectors = input.length;
	}
	
	public void deltaRule(){
		try {
			readPerceptronFile("perceptronTestData");
		} catch (IOException e) {
			System.out.println(e.getLocalizedMessage());
		}
		
		boolean allCorrect = false;
		int error = 0;
		while(!allCorrect) {
			int errorCount = 0;
			
			for(int i = 0; i < numberOfInputVectors; i++) {
				int result = 0;
				if(sumFunction(input[i]) > threshold) {
					result = 1;
				}
				error = output[i] - result;
				System.out.println(error);
				if(error != 0){
					errorCount += 1;
					for(int j = 0; j < sizeOfInputVectors; j++) {
						weights[j] += learningRate * error * input[i][j];
					}
				}
			}
			
			if(errorCount == 0){
				allCorrect = true;
			}
		}
	}
	
	public static void main(String[] args){
		Perceptron p = new Perceptron();
		p.deltaRule();
		
	}
	
}
