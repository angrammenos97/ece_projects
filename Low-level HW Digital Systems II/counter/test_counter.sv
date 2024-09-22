module test_dut;
	bit rst, ld_cnt, updn_cnt, count_enb, clk;
	bit [15:0] data_out, data_in;

	counter dut(.*);

	bind dut counter_property dutbound(.*);
	
	always #10 clk = !clk;

	initial
	begin
	clk = 0;				
	ld_cnt = 1;
	count_enb = 1;
	updn_cnt = 1;			
	rst = 0;					//at 0tu
	#15 rst = 1;			//at 15tu
	#15 data_in = 5;	//at 30tu
	#35 ld_cnt = 0; 	//at 65tu
	#10 ld_cnt = 1;		//at 75tu
	#10 updn_cnt = 0; //at 85tu
	#20 count_enb = 0;//at 105tu
	end

endmodule
