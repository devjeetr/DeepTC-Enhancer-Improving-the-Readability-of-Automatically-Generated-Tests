package main;

public class Calculator{
	private String name;

	Calculator() {
		name = "A calculator";
	}

	public int sum(int a, int b) {
		return a + b;
	}

	public void setName(String othername) {
		this.name = othername;
	}

}
