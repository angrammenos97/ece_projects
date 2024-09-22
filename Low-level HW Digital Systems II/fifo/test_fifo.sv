`define WIDTH 16
`define DEPTH 4

module test_fifo;
logic [`WIDTH-1:0] fifo_data_in;
logic rst, fifo_write,fifo_read,clk;
logic [`WIDTH-1:0] fifo_data_out;
logic fifo_full,fifo_empty;

fifo #(`WIDTH,`DEPTH) dut (.*);

bind dut fifo_property #(DEPTH) dutbound (clk,rst,fifo_write,fifo_read,fifo_full,fifo_empty,dut.wr_ptr,dut.rd_ptr,dut.cnt);

always #5 clk = !clk;

initial
begin
	clk = 0;
	fifo_data_in = 0;
	rst = 1;
	fifo_write = 0;
	fifo_read = 0;
	#2 rst = 0;						//at 2tu
	#2 rst = 1;						//at 4tu
	#6 fifo_data_in = 5;	//at 10tu
			fifo_write = 1;
	#10 fifo_data_in = 4;	//at 20tu
	#10 fifo_data_in = 3;	//at 30tu
	#10 fifo_data_in = 2;	//at 40tu
	#10 fifo_data_in = 1;	//at 50tu
	#10 fifo_read = 1;		//at 60tu
			fifo_write = 0;
	#10 fifo_read = 0;		//at 70tu
			fifo_write = 1;
			fifo_data_in = 1;
	#10 fifo_read = 1;		//at 80tu
			fifo_write = 0;
end

endmodule
