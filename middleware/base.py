from pandasai.middlewares import ChartsMiddleware


class CustomChartsMiddleware(ChartsMiddleware):
    def run(self, code: str) -> str:
        # code = super().run(code)

        processed = []
        for line in code.split("\n"):
            if line.find("plt.close()") != -1:
                idx = line.find("plt")
                blank = "".join([' ' for c in range(idx)])
                # Fix the chinese character display issue
                processed.append(blank + "plt.rcParams['font.sans-serif']=['SimHei']")
                processed.append(blank + "plt.rcParams['axes.unicode_minus']=False")
                # processed.append(blank + "plt.savefig('temp_chart.png')")
                processed.append(line)
            else:
                processed.append(line)
        code = "\n".join(processed)
        return code
