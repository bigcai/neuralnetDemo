package org.bigcai.vo;

import org.beetl.core.Configuration;
import org.beetl.core.GroupTemplate;
import org.beetl.core.Template;
import org.beetl.core.resource.StringTemplateResourceLoader;
import org.bigcai.util.HistoryRecorder;
import org.bigcai.util.vo.HistoryRecord;

import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ChartRender {

    private static Map<String, List<HistoryRecord>> readTemplateValues() {
        Map<String, List<HistoryRecord>> map = new HashMap<>();
        map.put("weightVectorHistory", HistoryRecorder.weightVectorHistory);
        return map;
    }


    /**
     * 渲染模板值到 neuralnetDemo\build\classes\java\test\chart.html
     * @throws Exception
     */
    public static void beetlString() throws Exception{
        // 读取模板
        InputStream inStream = ChartRender.class.getClassLoader().getResourceAsStream("chart.html.beetl");
        assert inStream != null;
        int length = inStream.available();
        byte[] buffer = new byte[length];
        inStream.read(buffer);
        inStream.close();
        String beetlChart = new String(buffer, StandardCharsets.UTF_8);

        // 初始化模板资源加载器
        StringTemplateResourceLoader resourceLoader = new StringTemplateResourceLoader();
        // 配置Beetl，这里使用默认配置
        Configuration config = Configuration.defaultConfiguration();
        // 初始化Beetl的核心GroupTemplate
        GroupTemplate groupTemplate = new GroupTemplate(resourceLoader, config);
        // 通过GroupTemplate传入自定义模板加载出Beetl模板Template
        Template template = groupTemplate.getTemplate(beetlChart);
        // 使用Template中的操作，将数据与占位符绑定

        Map<String, List<HistoryRecord>> map = readTemplateValues();

        template.binding(map);
        // 渲染字符串
        String str = template.render();
        System.out.println(str);

        File echartResult = new File(ChartRender.class.getClassLoader().getResource("").getPath()+"/chart.html");
        System.out.println(echartResult.getAbsoluteFile());

        echartResult.createNewFile();
        FileWriter fileWriter = new FileWriter(echartResult);
        fileWriter.write(str);
        fileWriter.flush();
        fileWriter.close();

    }

    public static void main(String[] args) throws Exception {

        beetlString();
    }
}

