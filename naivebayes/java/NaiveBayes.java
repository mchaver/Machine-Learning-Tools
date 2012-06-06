package naivebayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ListIterator;
import java.util.ArrayList;

public class NaiveBayes {
	private HashMap<String, Integer> SetA = new HashMap<String, Integer>();
	private HashMap<String, Integer> SetB = new HashMap<String, Integer>();	
	private Map<String, Double>Probabilities = new HashMap<String, Double>();

	private int ATokenWeight = 1;
	private int BTokenWeight = 1;
	private int TotalA;
	private int TotalB;
	private double MinCountForInclusion = 1;
	private double MinTokenProbability = 0.0001;
	private double MaxTokenProbability = 0.999;
	
	public void LoadTokens(List<String> listATokens, List<String> listBTokens){
		ListIterator<String> listATokensIterator = listATokens.listIterator();
		while (listATokensIterator.hasNext()) {
			String localToken = listATokensIterator.next();
			if(SetA.containsKey(localToken)) {
				SetA.put(listATokensIterator.next(), SetA.get(localToken)+1);
			} else {
				SetA.put(listATokensIterator.next(), 1);
			}
		}
		
		ListIterator<String> listBTokensIterator = listBTokens.listIterator();
		while(listBTokensIterator.hasNext()) {
			String localToken = listBTokensIterator.next();
			if(SetB.containsKey(localToken)) {
				SetB.put(listBTokensIterator.next(), SetB.get(localToken)+1);
			} else {
				SetB.put(listBTokensIterator.next(), 1);
			}
		}
		
		Probabilities = new HashMap<String, Double>();
		
		for(String token : SetA.keySet()){
			Probabilities.put(token, CalculateProbabilityOfToken(token));
		}
		for(String token : SetB.keySet()){
			if(!Probabilities.containsKey(token)) {
				Probabilities.put(token, CalculateProbabilityOfToken(token));
			}
		}
		
	}
	
	public double CalculateProbabilityOfTokens(List<String> tokens) {
		List<Double> tokenProbabilityList = new ArrayList<Double>();
		ListIterator<String> tokensIterator = tokens.listIterator();
		while(tokensIterator.hasNext()){
			double localTokenProbability = 0.5;
			String localToken = tokensIterator.next();
			if (Probabilities.containsKey(localToken)) {
				localTokenProbability = Probabilities.get(localToken);
				System.out.println(Probabilities.get(localToken));
			}
			tokenProbabilityList.add(localTokenProbability);
		}
		
		double totalProbability = 1;
		double negativeTotalProbability = 1;
		
		ListIterator<Double> tokenProbabilityListIterator = tokenProbabilityList.listIterator();
		while(tokenProbabilityListIterator.hasNext()){
			double localTokenProbability = tokenProbabilityListIterator.next();
			totalProbability *= localTokenProbability;
			negativeTotalProbability *= (1 - localTokenProbability);
		}
		return totalProbability / (totalProbability + negativeTotalProbability);
	}
	
	private double CalculateProbabilityOfToken(String token){
		double Probability = 0.0;
		int ACount = SetA.containsKey(token) ? SetA.get(token) * ATokenWeight : 0;
		int BCount = SetB.containsKey(token) ? SetB.get(token) * BTokenWeight : 0;
		if(ACount + BCount >= MinCountForInclusion) {
			double AProbability = Math.min(1,(double)ACount/(double)TotalA);
			double BProbability = Math.min(1, (double)BCount/(double)TotalB);
			Probability = Math.max(MinTokenProbability,Math.min(MaxTokenProbability, AProbability/(AProbability + BProbability)));
		}
		return Probability;
	}
}
